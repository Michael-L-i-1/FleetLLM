// Model-free white-box tests of the actual server reply, routing, cancellation
// and telemetry functions. Link the CPU library; server.c is included so no
// production testing API or fake network endpoint is needed.
#include "../src/server.c"
#define CHATCHECK_NO_MAIN
#include "chatglimmercheck.c"
#include <assert.h>

static req *new_request(void) {
    req *r=calloc(1,sizeof *r);
    pthread_mutex_init(&r->mu,NULL);pthread_cond_init(&r->cv,NULL);
    atomic_init(&r->cancelled,false);
    r->tools=true;r->finish="stop";
    r->tool_names=calloc(1,sizeof(char *));r->tool_names[0]=strdup("write");r->n_tool_names=1;
    return r;
}

static json *reply(server *s,req *r,const char *finish) {
    buf b={0};bstr(&b,"{");
    expect_str(reply_fields(s,r,&b),finish,"response-finish-reason");bstr(&b,"}");
    json *j=json_parse(b.s,b.n);expect(j!=NULL,"reply-is-json");free(b.s);return j;
}

static void test_replies(void) {
    server s={.fam=CHAT_GLIMMER};req *r=new_request();
    for(int i=0;i<12;i++) {bstr(&r->text[CH_TOOL],"<|start|>assistant to=write<|message|>");bstr(&r->text[CH_TOOL],CALL);bstr(&r->text[CH_TOOL],"<|eom|>");}
    json *j=reply(&s,r,"tool_calls");const jnode *calls=json_get(j,json_root(j),"tool_calls");
    expect(calls && calls->n==12,"server-preserves-twelve-calls");
    if(calls) for(int i=0;i<calls->n;i++) {
        const jnode *c=json_child(j,calls,i),*fn=json_get(j,c,"function");
        expect_str(json_str(j,fn,"name"),"write","server-registered-name-binding");
        for(int k=0;k<i;k++) expect(strcmp(json_str(j,c,"id"),json_str(j,json_child(j,calls,k),"id"))!=0,"unique-call-ids");
        expect(strlen(json_str(j,c,"id"))>20,"call-id-instance-nonce");
    }
    json_free(j);
    bstr(&r->text[CH_TOOL],"<|start|>assistant to=write<|message|><atem:invoke name=\"write\"><atem:parameter name=\"content\">truncated");
    r->finish="length";j=reply(&s,r,"length");
    expect(!json_get(j,json_root(j),"tool_calls"),"cap-discards-even-complete-earlier-calls");json_free(j);
    r->finish="cancelled";r->cancelled=true;j=reply(&s,r,"cancelled");
    expect(!json_get(j,json_root(j),"tool_calls"),"cancel-no-executable-fields");json_free(j);free_req(r);
    r=new_request();bstr(&r->text[CH_CONTENT],CALL);bstr(&r->text[CH_REASONING],CALL);
    j=reply(&s,r,"stop");expect(!json_get(j,json_root(j),"tool_calls"),"echoed-markup-not-executable");json_free(j);free_req(r);
}

static void test_routing(void) {
    const char *parts[]={"<|message|>","<|start|>","<|eom|>","<|eot|>","<|end_of_text|>"," to=self","assistant to=write","assistant to=user",CALL,"answer"};
    tokenizer *t=make_tok("routing",parts,10,NULL);
    server s={.fam=CHAT_GLIMMER,.tok=t,.t_start_id=tok_lookup(t,parts[1]),.t_message=tok_lookup(t,parts[0]),.t_eom=tok_lookup(t,parts[2]),.t_eot=tok_lookup(t,parts[3]),.t_eos=tok_lookup(t,parts[4])};
    req *r=new_request();r->pstate=PS_HEADER;
    int sequence[]={5,0,8,2,1,6,0,8,2,1,7,0,9,3};
    for(size_t i=0;i<sizeof sequence/sizeof *sequence;i++) route_token(&s,r,tok_lookup(t,parts[sequence[i]]));
    expect_str(r->text[CH_CONTENT].s,"answer","routing-user-content");
    expect_str(r->text[CH_REASONING].s,CALL,"routing-self-echo");
    expect_has(r->text[CH_TOOL].s,"<|start|>assistant to=write<|message|>","routing-retains-tool-framing");
    json *j=reply(&s,r,"tool_calls");const jnode *calls=json_get(j,json_root(j),"tool_calls");
    expect(calls && calls->n==1,"routed-exactly-one-real-call");json_free(j);free_req(r);
    r=new_request();r->pstate=PS_MESSAGE;route_token(&s,r,tok_lookup(t,"answer"));
    expect_str(r->text[CH_CONTENT].s,"answer","direct-prefix-content-not-swallowed");free_req(r);
    r=new_request();out_bytes(r,CH_CONTENT,"\xf0",1);
    expect(!r->text[CH_CONTENT].n,"utf8-leading-byte-buffered");
    out_bytes(r,CH_CONTENT,"\x9f\x98",2);expect(!r->text[CH_CONTENT].n,"utf8-incomplete-tail-buffered");
    out_bytes(r,CH_CONTENT,"\x80",1);expect_str(r->text[CH_CONTENT].s,"\xf0\x9f\x98\x80","utf8-split-codepoint-restored");free_req(r);
    tok_free(t);
}

typedef struct {req *r;int fd;} waiter;
static void *wait_request(void *arg) {waiter *w=arg;wait_done(w->r,w->fd);return NULL;}

static void test_cancellation(void) {
    int sockets[2];assert(!socketpair(AF_UNIX,SOCK_STREAM,0,sockets));
    req *first=new_request(),*cancel=new_request(),*last=new_request();first->next=cancel;cancel->next=last;
    server_opts opts={.max_active=1};server s={.o=&opts,.queue=first,.n_active=1};
    pthread_mutex_init(&s.mu,NULL);
    waiter w={cancel,sockets[0]};pthread_t th;assert(!pthread_create(&th,NULL,wait_request,&w));
    expect(!fleet_peer_gone(sockets[0]),"live-peer");
    close(sockets[1]);
    double start=wall();
    while(!atomic_load(&cancel->cancelled) && wall()-start<2) {struct timespec delay={0,1000000};nanosleep(&delay,NULL);}
    expect(atomic_load(&cancel->cancelled),"buffered-disconnect-detected");
    pthread_mutex_lock(&s.mu);drain_cancelled(&s);pthread_mutex_unlock(&s.mu);
    pthread_join(th,NULL);
    expect(cancel->done && !cancel->started,"cancel-acked-before-admission");
    expect(s.queue==first && first->next==last && s.n_active==1,"cancel-drained-with-all-slots-busy");
    expect(wall()-start<1,"cancel-latency-under-one-second");
    // Ack is the ownership boundary: freeing here must not invalidate queue links.
    free_req(cancel);expect(s.queue->next==last,"queue-survives-cancelled-free");
    first->cancelled=true;last->cancelled=true;
    pthread_mutex_lock(&s.mu);drain_cancelled(&s);pthread_mutex_unlock(&s.mu);
    expect(!s.queue && first->done && last->done,"drain-all-cancelled-positions");
    free_req(first);free_req(last);close(sockets[0]);pthread_mutex_destroy(&s.mu);
}

static json *endpoint(server *s,const char *path) {
    int sockets[2];assert(!socketpair(AF_UNIX,SOCK_STREAM,0,sockets));
    handle_fleet(s,sockets[0],path);shutdown(sockets[0],SHUT_WR);
    char response[8192];size_t n=0;ssize_t got;
    while((got=recv(sockets[1],response+n,sizeof response-n-1,0))>0) n+=(size_t)got;
    response[n]=0;char *body=strstr(response,"\r\n\r\n");assert(body);
    json *j=json_parse(body+4,strlen(body+4));close(sockets[0]);close(sockets[1]);return j;
}

static void test_stats(void) {
    const char *parts[]={"<|message|>"};tokenizer *t=make_tok("stats",parts,1,NULL);
    engine e={0};e.kv.full.n_pages=100;e.kv.full.n_free=25;e.kv.full.page_stride=16;e.kv.full.elem=4;e.kv.full.bytes=6400;
    e.st.shared_tokens=123;e.st.prefill_rows=200;
    server_opts opts={.max_active=8,.model_id="test",.admit=true};
    server s={.e=&e,.tok=t,.fam=CHAT_GLIMMER,.o=&opts,.n_active=3,.served=4,.gen_tokens=50,.prompt_tokens=300,.t_start=now_secs()};
    pthread_mutex_init(&s.mu,NULL);snap_engine(&s);
    // HTTP readers must use the snapshot, not mutable engine/thread counters.
    s.n_active=99;s.served=999;e.kv.full.n_free=0;
    json *j=endpoint(&s,"/v1/fleet/stats");const jnode *root=json_root(j),*kv=json_get(j,root,"kv");
    expect(json_num(j,root,"active",-1)==3,"stats-active-snapshot");
    expect(json_num(j,json_get(j,root,"requests"),"completed",-1)==4,"stats-completed-snapshot");
    expect(json_num(j,kv,"total_bytes",-1)==6400 && json_num(j,kv,"free_bytes",-1)==1600,"stats-physical-kv-bytes");
    expect(json_get(j,json_get(j,root,"memory"),"physical_footprint_bytes")->type==J_NULL,"footprint-not-invented");json_free(j);
    j=endpoint(&s,"/v1/fleet/capabilities");root=json_root(j);
    expect(json_num(j,root,"max_active",-1)==8,"capability-max-active");
    expect(json_get(j,root,"context_window")->type==J_NULL,"unknown-model-context-not-kv-capacity");
    expect_str(json_str(j,json_get(j,root,"features"),"tool_streaming"),"buffered","honest-tool-streaming-capability");json_free(j);
    pthread_mutex_destroy(&s.mu);tok_free(t);
}

int main(void) {
    test_replies();test_routing();test_cancellation();test_stats();
    printf("Server reply/routing/cancellation/telemetry checks: %s\n",failures?"FAILED":"passed");
    return failures?1:0;
}
