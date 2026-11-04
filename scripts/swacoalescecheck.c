// Actual engine scheduling with a tiny deterministic full+SWA fake backend.
// No model weights, downloads, GPU, or approximations. See docs/swa-coalescing.md.
#include "engine.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VOCAB 32
#define OUT 4
#define MAXPROMPT 6120
static long cows;
static void *allocate(backend *b, size_t n, void **handle) { return *handle = calloc(1, n ? n : 1); }
static float load(kv_pool *p, uint32_t page, int pos) {
    assert(page != KV_NONE);
    float x; kv_vec_load(p, page, 0, 0, 0, pos % KV_PAGE, &x); return x;
}
static unsigned hash(unsigned h, int token, int pos) { return (h * 7u + (unsigned)token + (unsigned)pos) % VOCAB; }
static int forward(backend *b, const model *m, kv_cache *kv, const seq_pages *const *pages, const row_t *rows, int n, float *logits) {
    int out = 0;
    for (int i = 0; i < n; i++) {
        const row_t *r = rows + i;
        const seq_pages *s = pages[r->seq];
        // Every required window page must exist at each sequence's first row.
        if (!i || rows[i-1].seq != r->seq)
            for (int pos = r->pos > kv->window ? r->pos - kv->window + 1 : 0; pos < r->pos; pos++)
                assert(s->swa[pos / KV_PAGE] != KV_NONE);
        unsigned h = r->pos ? (unsigned)load(&kv->full, s->full[(r->pos-1) / KV_PAGE], r->pos-1) : 0;
        float state = (float)hash(h, r->token, r->pos), token = (float)r->token;
        kv_vec_store(&kv->full, s->full[r->pos / KV_PAGE], 0, 0, 0, r->pos % KV_PAGE, &state);
        kv_vec_store(&kv->swa, s->swa[r->pos / KV_PAGE], 0, 0, 0, r->pos % KV_PAGE, &token);
        if (r->logits) {
            if (b->argmax) b->argmax[out] = (int)state;
            if (logits) for (int j=0;j<VOCAB;j++) logits[out*VOCAB+j] = -(float)abs(j-(int)state);
            out++;
        }
    }
    return 0;
}
static model fixture(void) {
    model m = {0};
    m.c = (model_cfg){.arch=ARCH_GLIMMER,.n_layer=2,.n_full=1,.n_swa=1,.n_head=1,.n_kv=1,.hd=1,.d=1,.vocab=VOCAB,.window=2048};
    return m;
}
static backend compute(void) { return (backend){.name="tiny SWA reference",.kv_elem=4,.alloc_kv=allocate,.forward=forward,.argmax_ok=true}; }
static sample_params params(void) {
    sample_params p=sample_defaults(); p.temperature=0;p.max_new=OUT;p.ignore_eos=true;p.n_stop=1;p.stop[0]=-1;return p;
}
static void finish(engine *e,const int *ids,int n) {
    int steps=0;
    for (;;) {
        int active[8], count=0;
        for(int i=0;i<n;i++) if(e->seqs[ids[i]].live && !e->seqs[ids[i]].done) active[count++]=ids[i];
        if(!count) break;
        assert(++steps<4000);
        assert(engine_step(e,active,count));
    }
}
static void cleanup(engine *e) {
    for(int i=0;i<e->n_seqs;i++) if(e->seqs[i].live) engine_release(e,i);
    assert(e->kv.full.n_free==e->kv.full.n_pages);
    assert(e->kv.swa.n_free==e->kv.swa.n_pages);
    void *f=e->kv.full.data,*w=e->kv.swa.data,*fr=e->kv.full.refcnt,*wr=e->kv.swa.refcnt,*ff=e->kv.full.free_list,*wf=e->kv.swa.free_list;
    engine_free(e);free(f);free(w);free(fr);free(wr);free(ff);free(wf);
}
static void verify(engine *e,int id,const int32_t *prompt,int len) {
    seq_t *s=&e->seqs[id];unsigned h=0;
    for(int i=0;i<len;i++) h=hash(h,prompt[i],i);
    assert(s->n_gen==OUT);
    for(int i=0;i<OUT;i++) {assert(s->gen[i]==(int)h);h=hash(h,(int)h,len+i);}
    // Verify retained SWA values, including generated tokens, against history.
    for(int pos=s->pages.len-e->kv.window+1;pos<s->pages.len;pos++) if(pos>=0) {
        float got=load(&e->kv.swa,s->pages.swa[pos/KV_PAGE],pos);
        assert(got==(float)s->tokens[pos]);
    }
}
static void make_prompt(int32_t *p,int len,int prefix,int branch) {
    for(int i=0;i<len;i++) p[i]=i<prefix ? (i*11+3)%VOCAB : (i*11+3+branch)%VOCAB;
}
static long fanout(bool enabled,int prefix,bool staggered) {
    model m=fixture();backend b=compute();engine *e=engine_new(&m,&b,NULL,2u<<20);
    e->recurrent_prefix_dedup=enabled;e->row_budget=128;
    int32_t prompts[8][MAXPROMPT];int ids[8],lengths[8];long expected=0;
    for(int i=0;i<8;i++) {
        int shared=staggered && i ? prefix-i*13 : prefix;
        lengths[i]=prefix+16;
        make_prompt(prompts[i],lengths[i],shared,i);
        ids[i]=engine_create(e,prompts[i],lengths[i]);engine_start(e,ids[i],params());
        if(i) expected+=shared;
    }
    finish(e,ids,8);
    for(int i=0;i<8;i++) verify(e,ids[i],prompts[i],lengths[i]);
    long reused=e->st.shared_tokens;
    if(enabled) {assert(reused==expected);for(int i=1;i<8;i++) assert(e->seqs[ids[i]].prefix_cached==(staggered?prefix-i*13:prefix));}
    if(!enabled && prefix==6000 && !staggered) assert(reused==0);
    printf("fanout enabled=%d prefix=%d staggered=%d shared=%ld prefill=%ld\n",enabled,prefix,staggered,reused,e->st.prefill_rows);
    cleanup(e);return reused;
}
static void boundary_and_cow(void) {
    model m=fixture();backend b=compute();engine *e=engine_new(&m,&b,NULL,2u<<20);
    e->recurrent_prefix_dedup=true;e->row_budget=128;
    int32_t p[2][6016];int ids[2];
    for(int i=0;i<2;i++) {make_prompt(p[i],6016,6000,i);ids[i]=engine_create(e,p[i],6016);engine_start(e,ids[i],params());}
    while(e->seqs[ids[0]].n_tokens<6000) assert(engine_step(e,ids,2));
    assert(e->seqs[ids[0]].n_tokens==6000);
    assert(e->seqs[ids[0]].pages.swa[247]!=KV_NONE);
    // A one-token planner budget forks child at start of next step then only
    // advances leader. Shared full/SWA tail page must be copied before writing.
    e->row_budget=1;assert(engine_step(e,ids,2));
    seq_t *a=&e->seqs[ids[0]],*c=&e->seqs[ids[1]];
    assert(c->prefix_cached==6000 && c->n_tokens==6000);
    assert(a->pages.full[374]==c->pages.full[374]); // boundary page6000/16=375, prior page374 remains shared
    assert(e->kv.full.refcnt[a->pages.full[374]]==2);
    // 6000 is page-aligned: use unaligned case separately below.
    finish(e,ids,2);verify(e,ids[0],p[0],6016);verify(e,ids[1],p[1],6016);cleanup(e);
    e=engine_new(&m,&b,NULL,2u<<20);e->recurrent_prefix_dedup=true;e->row_budget=128;
    for(int i=0;i<2;i++) {make_prompt(p[i],6016,6003,i);ids[i]=engine_create(e,p[i],6016);engine_start(e,ids[i],params());}
    while(e->seqs[ids[0]].n_tokens<6003) assert(engine_step(e,ids,2));
    e->row_budget=1;assert(engine_step(e,ids,2));
    a=&e->seqs[ids[0]];c=&e->seqs[ids[1]];
    assert(c->n_tokens==6003 && c->prefix_cached==6003);
    assert(a->pages.full[375]!=c->pages.full[375]);
    assert(a->pages.swa[375]!=c->pages.swa[375]);cows++;
    finish(e,ids,2);verify(e,ids[0],p[0],6016);verify(e,ids[1],p[1],6016);cleanup(e);
}
static void missing_leader(bool release) {
    model m=fixture();backend b=compute();engine *e=engine_new(&m,&b,NULL,2u<<20);
    e->recurrent_prefix_dedup=true;
    int32_t p[2][6016];int ids[2];
    for(int i=0;i<2;i++) {make_prompt(p[i],6016,6000,i);ids[i]=engine_create(e,p[i],6016);engine_start(e,ids[i],params());}
    assert(e->seqs[ids[1]].wait_len==6000);
    if(release) engine_release(e,ids[0]); // cancellation / leader disappearance
    finish(e,ids+1,1); // otherwise omitted leader triggers existing forced fallback
    assert(e->seqs[ids[1]].prefix_cached==0);
    verify(e,ids[1],p[1],6016);cleanup(e);
}
static void cancelled_waiter(void) {
    model m=fixture();backend b=compute();engine *e=engine_new(&m,&b,NULL,2u<<20);
    e->recurrent_prefix_dedup=true;e->row_budget=128;
    int32_t p[2][6016];int ids[2];
    for(int i=0;i<2;i++) {make_prompt(p[i],6016,6000,i);ids[i]=engine_create(e,p[i],6016);engine_start(e,ids[i],params());}
    assert(e->seqs[ids[1]].wait_len==6000);
    engine_retire(e,ids[1]);
    assert(e->seqs[ids[1]].wait_len==0);
    while(e->seqs[ids[0]].n_tokens<6000) assert(engine_step(e,ids,1));
    assert(e->seqs[ids[0]].n_tokens==6016); // canceled child no longer imposes boundary
    finish(e,ids,1);verify(e,ids[0],p[0],6016);cleanup(e);
}
static void memory_pressure(void) {
    model m=fixture();backend b=compute();engine *e=engine_new(&m,&b,NULL,240000);
    e->recurrent_prefix_dedup=true;e->row_budget=128;
    int32_t p[2][6016];int ids[2];bool finished[2]={false,false};
    for(int i=0;i<2;i++) {make_prompt(p[i],6016,3000,i);ids[i]=engine_create(e,p[i],6016);engine_start(e,ids[i],params());}
    int steps=0;
    while(!finished[0] || !finished[1]) {
        int active[2],n=0;
        for(int i=0;i<2;i++) if(!finished[i]) active[n++]=ids[i];
        assert(++steps<4000 && engine_step(e,active,n));
        for(int i=0;i<2;i++) if(!finished[i] && e->seqs[ids[i]].done) {
            verify(e,ids[i],p[i],6016);finished[i]=true;engine_retire(e,ids[i]);
        }
    }
    assert(!e->seqs[ids[0]].live); // cached leader was evicted to fit divergent suffix
    printf("pressure cached-leader-evicted=1 preemptions=%ld shared=%ld\n",e->st.preemptions,e->st.shared_tokens);
    cleanup(e);
}

// An older, longer textual match has already discarded its required SWA
// pages. It must not hide a younger usable source or a live future endpoint.
static void stale_leader(bool retired, bool future) {
    model m=fixture();backend b=compute();engine *e=engine_new(&m,&b,NULL,2u<<20);
    e->recurrent_prefix_dedup=true;e->row_budget=128;
    int32_t old[6016],fresh[6016],child[6016];
    make_prompt(old,6016,6016,0);make_prompt(fresh,6016,2500,1);make_prompt(child,6016,3000,2);
    int stale=engine_create(e,old,6016);engine_start(e,stale,params());finish(e,&stale,1);
    assert(!kv_can_fork_at(&e->kv,&e->seqs[stale].pages,3000));
    if(retired) engine_retire(e,stale);
    int active_len=future?6016:2500;
    int active=engine_create(e,fresh,active_len);engine_start(e,active,params());
    if(!future) {
        e->row_budget=100;
        while(e->seqs[active].n_tokens<2500) assert(engine_step(e,&active,1));
        assert(e->seqs[active].n_tokens==2500);
        assert(kv_can_fork_at(&e->kv,&e->seqs[active].pages,2500));
    }
    int id=engine_create(e,child,6016);engine_start(e,id,params());
    if(future) assert(e->seqs[id].wait_len==2500 && e->seqs[id].wait_for==active);
    else assert(!e->seqs[id].wait_len && e->seqs[id].prefix_cached==2500);
    int ids[]={active,id};finish(e,ids,2);
    assert(e->seqs[id].prefix_cached==2500);
    verify(e,active,fresh,active_len);verify(e,id,child,6016);
    printf("stale leader retired=%d future=%d reused=2500\n",retired,future);cleanup(e);
}
int main(void) {
    assert(fanout(false,6000,false)==0);
    assert(fanout(true,6000,false)==7*6000);
    fanout(false,6016,false);fanout(true,6016,false);
    fanout(true,6003,false);fanout(true,6080,true);
    boundary_and_cow();missing_leader(true);missing_leader(false);cancelled_waiter();memory_pressure();
    stale_leader(false,false);stale_leader(false,true);stale_leader(true,false);stale_leader(true,true);
    assert(cows==1);
    puts("SWA scheduling, output/KV parity, COW and release checks passed");
}
