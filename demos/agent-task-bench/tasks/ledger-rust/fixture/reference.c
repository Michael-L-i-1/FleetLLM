/* Legacy Ledger CLI. Port its behavior into the eight Rust modules. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>
#define MAX_ROWS 2048
#define TEXT 4096
#define FIELDS 32
typedef struct { char id[TEXT], date[TEXT], vendor[TEXT]; int64_t cents; } Row;
static int space(unsigned char c) { return c==' '||c=='\t'||c=='\n'||c=='\r'||c=='\v'||c=='\f'; }
static unsigned char lower(unsigned char c) { return c>='A'&&c<='Z'?c+32:c; }
static void json(const char *s) {
    putchar('"'); for (const unsigned char *p=(const unsigned char*)s;*p;p++) {
        if(*p=='"'||*p=='\\') { putchar('\\');putchar(*p); }
        else if(*p<32) printf("\\u%04x",*p); else putchar(*p);
    } putchar('"');
}
/* money.rs */
static int money_parse(const char *s,int64_t *out) {
    int neg=0; if(*s=='-'||*s=='+') {neg=*s=='-';s++;}
    if(*s<'0'||*s>'9') return 0;
    uint64_t whole=0, limit=neg?UINT64_C(9223372036854775808):INT64_MAX;
    while(*s>='0'&&*s<='9') {unsigned d=*s++-'0';if(whole>(limit/100-d)/10)return 0;whole=whole*10+d;}
    unsigned frac=0;
    if(*s=='.') {s++;if(*s<'0'||*s>'9')return 0;frac=(*s++-'0')*10;if(*s>='0'&&*s<='9')frac+=*s++-'0';}
    if(*s || whole>limit/100 || (whole==limit/100&&frac>limit%100))return 0;
    uint64_t n=whole*100+frac;
    *out=neg?(n==UINT64_C(9223372036854775808)?INT64_MIN:-(int64_t)n):(int64_t)n;return 1;
}
static void money_format(int64_t n,char *out) {
    uint64_t a=n<0?(uint64_t)(-(n+1))+1:(uint64_t)n;
    sprintf(out,"%s%llu.%02llu",n<0?"-":"",(unsigned long long)(a/100),(unsigned long long)(a%100));
}
/* date.rs */
static int date_valid(const char *s) {
    if(strlen(s)!=10||s[4]!='-'||s[7]!='-')return 0;
    for(int i=0;i<10;i++)if(i!=4&&i!=7&&(s[i]<'0'||s[i]>'9'))return 0;
    int y=(s[0]-'0')*1000+(s[1]-'0')*100+(s[2]-'0')*10+s[3]-'0';
    int m=(s[5]-'0')*10+s[6]-'0',d=(s[8]-'0')*10+s[9]-'0';
    int days[]={0,31,28,31,30,31,30,31,31,30,31,30,31};
    if(y<1||m<1||m>12)return 0;if(y%4==0&&(y%100!=0||y%400==0))days[2]=29;
    return d>=1&&d<=days[m];
}
/* csv.rs */
static int csv_parse(const char *s,char fields[FIELDS][TEXT]) {
    if(strchr(s,'\r')||strchr(s,'\n'))return -1;
    int count=0;while(1) {
        if(count==FIELDS)return -1;int n=0;
        if(*s=='"') {s++;while(1) {if(!*s)return -1;if(*s=='"') {s++;if(*s=='"'){if(n>=TEXT-1)return -1;fields[count][n++]='"';s++;}else break;}else {if(n>=TEXT-1)return -1;fields[count][n++]=*s++;}}if(*s&&*s!=',')return -1;}
        else {while(*s&&*s!=',') {if(*s=='"'||n>=TEXT-1)return -1;fields[count][n++]=*s++;}}
        fields[count++][n]=0;if(!*s)break;s++;
    }return count;
}
/* labels.rs */
static void normalize(const char *s,char *out) {
    int n=0,pending=0;for(;*s;s++) {if(space((unsigned char)*s)){if(n)pending=1;}else {if(pending)out[n++]=' ';pending=0;out[n++]=lower((unsigned char)*s);}}out[n]=0;
}
/* tags.rs */
static int cmp_text(const void *a,const void *b) {return strcmp(a,b);}
static int tags_parse(const char *s,char values[FIELDS][TEXT]) {
    int n=0;while(1) {const char *end=strchr(s,';');if(!end)end=s+strlen(s);while(s<end&&space((unsigned char)*s))s++;while(end>s&&space((unsigned char)end[-1]))end--;
        if(end>s) {if(n==FIELDS||end-s>=TEXT)return -1;int k=0;for(const char *p=s;p<end;p++){unsigned char c=lower((unsigned char)*p);if(!((c>='a'&&c<='z')||(c>='0'&&c<='9')||c=='_'||c=='-'))return -1;values[n][k++]=c;}values[n++][k]=0;}
        const char *sep=strchr(s,';');if(!sep)break;s=sep+1;
    }qsort(values,n,TEXT,cmp_text);int unique=0;for(int i=0;i<n;i++)if(!unique||strcmp(values[i],values[unique-1])){if(i!=unique)strcpy(values[unique],values[i]);unique++;}return unique;
}
static int load(const char *path,Row *rows,int *invalid) {
    FILE *f=fopen(path,"r");if(!f)return -1;char line[TEXT*5];int n=0;*invalid=0;fgets(line,sizeof(line),f);
    while(fgets(line,sizeof(line),f)) {line[strcspn(line,"\r\n")]=0;char fields[FIELDS][TEXT],ts[FIELDS][TEXT];int64_t amount;char vendor[TEXT];int count=csv_parse(line,fields);
        if(count!=5||!date_valid(fields[1])||!money_parse(fields[3],&amount)||tags_parse(fields[4],ts)<0){(*invalid)++;continue;}
        char *start=fields[0];while(space((unsigned char)*start))start++;char *end=start+strlen(start);while(end>start&&space((unsigned char)end[-1]))*--end=0;
        normalize(fields[2],vendor);if(!*start||!*vendor){(*invalid)++;continue;}
        if(n==MAX_ROWS){fclose(f);return -1;}strcpy(rows[n].id,start);strcpy(rows[n].date,fields[1]);strcpy(rows[n].vendor,vendor);rows[n++].cents=amount;
    }fclose(f);return n;
}
/* audit.rs: stable first-wins deduplication, unique sorted duplicate IDs. */
static int dedupe_rows(Row *rows,int n,char dup[MAX_ROWS][TEXT],int *ndup) {
    int out=0;*ndup=0;for(int i=0;i<n;i++) {int seen=0;for(int j=0;j<out;j++)if(!strcmp(rows[j].id,rows[i].id)){seen=1;break;}
        if(seen) {int found=0;for(int j=0;j<*ndup;j++)if(!strcmp(dup[j],rows[i].id))found=1;if(!found)strcpy(dup[(*ndup)++],rows[i].id);}
        else {if(i!=out)rows[out]=rows[i];out++;}
    }qsort(dup,*ndup,TEXT,cmp_text);return out;
}
/* filter.rs: dates have validated lexicographically sortable ISO form. */
static int filter_rows(Row *rows,int n,const char *from,const char *to) {
    if(!date_valid(from)||!date_valid(to)||strcmp(from,to)>0)return 0;
    int out=0;for(int i=0;i<n;i++)if(strcmp(rows[i].date,from)>=0&&strcmp(rows[i].date,to)<=0){if(out!=i)rows[out]=rows[i];out++;}return out;
}
/* summary.rs */
static int cmp_vendor(const void *a,const void *b) {return strcmp(((const Row*)a)->vendor,((const Row*)b)->vendor);}
static void group_rows(Row *rows,int n) {
    qsort(rows,n,sizeof(Row),cmp_vendor);putchar('[');int first=1;
    for(int i=0;i<n;) {int j=i;int64_t total=0;while(j<n&&!strcmp(rows[j].vendor,rows[i].vendor))total+=rows[j++].cents;
        if(!first)putchar(',');first=0;printf("{\"vendor\":");json(rows[i].vendor);printf(",\"count\":%d,\"total\":",j-i);char b[64];money_format(total,b);json(b);putchar('}');i=j;
    }putchar(']');
}
static void string_list(char rows[][TEXT],int n) {putchar('[');for(int i=0;i<n;i++){if(i)putchar(',');json(rows[i]);}putchar(']');}
static void ids(Row *rows,int n) {putchar('[');for(int i=0;i<n;i++){if(i)putchar(',');json(rows[i].id);}putchar(']');}
int main(int argc,char **argv) {
    if(argc<3)return 2;const char *cmd=argv[1],*v=argv[2];char fields[FIELDS][TEXT],b[TEXT];int count;int64_t cents;
    if(!strcmp(cmd,"money")){if(money_parse(v,&cents)){money_format(cents,b);puts(b);}else puts("ERR");return 0;}
    if(!strcmp(cmd,"date")){puts(date_valid(v)?v:"ERR");return 0;}
    if(!strcmp(cmd,"csv")){count=csv_parse(v,fields);if(count<0)puts("ERR");else{string_list(fields,count);puts("");}return 0;}
    if(!strcmp(cmd,"label")){normalize(v,b);puts(b);return 0;}
    if(!strcmp(cmd,"tags")){count=tags_parse(v,fields);if(count<0)puts("ERR");else{string_list(fields,count);puts("");}return 0;}
    const char *path=v;if(!strcmp(cmd,"filter")){if(argc<5)return 2;path=argv[4];}
    Row *rows=calloc(MAX_ROWS,sizeof(Row));char (*dup)[TEXT]=calloc(MAX_ROWS,TEXT);int invalid=0,ndup=0;int n=load(path,rows,&invalid);if(n<0)return 2;n=dedupe_rows(rows,n,dup,&ndup);
    if(!strcmp(cmd,"summary"))group_rows(rows,n);
    else if(!strcmp(cmd,"audit"))string_list(dup,ndup);
    else if(!strcmp(cmd,"filter")){if(!date_valid(v)||!date_valid(argv[3])||strcmp(v,argv[3])>0)printf("ERR");else ids(rows,filter_rows(rows,n,v,argv[3]));}
    else if(!strcmp(cmd,"report")){if(argc<5)return 2;if(!date_valid(argv[3])||!date_valid(argv[4])||strcmp(argv[3],argv[4])>0)printf("ERR");else{printf("{\"accepted\":%d,\"invalid\":%d,\"duplicates\":",n,invalid);string_list(dup,ndup);printf(",\"selected\":");n=filter_rows(rows,n,argv[3],argv[4]);ids(rows,n);printf(",\"totals\":");group_rows(rows,n);putchar('}');}}
    else return 2;puts("");free(rows);free(dup);return 0;
}
