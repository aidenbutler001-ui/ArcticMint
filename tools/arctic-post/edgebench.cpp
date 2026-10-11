// microbench: boundary-edge detection on a real(ish) 17M-face mesh: node-hash vs sort vs parallel sort
#include <cstdio>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <thread>
#include <atomic>
#include <chrono>
using namespace std;
static double now(){ return chrono::duration<double>(chrono::steady_clock::now().time_since_epoch()).count(); }
int main(int argc,char**argv){
  FILE*f=fopen(argv[1],"rb"); int h[4]; fread(h,4,4,f); int V=h[0],F=h[1];
  fseek(f,16+(long)V*12,SEEK_SET); vector<int32_t> faces((size_t)F*3); fread(faces.data(),4,faces.size(),f); fclose(f);
  auto ukey=[](int a,int b){ if(a>b) swap(a,b); return ((uint64_t)(uint32_t)a<<32)|(uint32_t)b; };
  double t=now(); size_t nb=0;
  { struct Use{int count=0;uint64_t d=0;}; unordered_map<uint64_t,Use> m; m.reserve((size_t)F*3);
    for(size_t i=0;i<(size_t)F;++i) for(int j=0;j<3;++j){ int a=faces[3*i+j],b=faces[3*i+(j+1)%3]; auto&e=m[ukey(a,b)]; if(++e.count==1) e.d=((uint64_t)a<<32)|b; }
    for(auto&kv:m) if(kv.second.count==1) ++nb; }
  printf("unordered_map (as fill_small_holes): %.2fs boundary=%zu\n",now()-t,nb);
  t=now(); vector<uint64_t> k((size_t)F*3);
  for(size_t i=0;i<(size_t)F;++i) for(int j=0;j<3;++j) k[3*i+j]=ukey(faces[3*i+j],faces[3*i+(j+1)%3]);
  sort(k.begin(),k.end()); nb=0; for(size_t i=0;i<k.size();){ size_t j=i; while(j<k.size()&&k[j]==k[i])++j; if(j-i==1)++nb; i=j; }
  printf("std::sort 1 thread: %.2fs boundary=%zu\n",now()-t,nb);
  // parallel: build keys in parallel, MSD partition by top bits of first vertex into P*8 buckets, sort buckets in parallel
  for(int T: {1,4}){
    t=now(); size_t n=(size_t)F*3; vector<uint64_t> kk(n);
    auto par=[&](auto fn){ vector<thread> th; for(int w=0;w<T;++w) th.emplace_back(fn,w); for(auto&x:th) x.join(); };
    par([&](int w){ size_t b=n/3*w/T, e=n/3*(w+1)/T; for(size_t i=b;i<e;++i) for(int j=0;j<3;++j) kk[3*i+j]=ukey(faces[3*i+j],faces[3*i+(j+1)%3]); });
    const int NBK=256; const int shift=32+ (V>1 ? 0:0);
    // bucket = (smaller vertex id * NBK / V)
    vector<vector<size_t>> cnt(T, vector<size_t>(NBK,0));
    par([&](int w){ size_t b=n*w/T, e=n*(w+1)/T; for(size_t i=b;i<e;++i) ++cnt[w][(kk[i]>>32)*NBK/V]; });
    vector<size_t> bstart(NBK+1,0); vector<vector<size_t>> off(T, vector<size_t>(NBK));
    size_t pos=0; for(int bk=0;bk<NBK;++bk){ bstart[bk]=pos; for(int w=0;w<T;++w){ off[w][bk]=pos; pos+=cnt[w][bk]; } } bstart[NBK]=pos;
    vector<uint64_t> out(n);
    par([&](int w){ size_t b=n*w/T, e=n*(w+1)/T; for(size_t i=b;i<e;++i) out[off[w][(kk[i]>>32)*NBK/V]++]=kk[i]; });
    atomic<int> next{0}; vector<size_t> nbw(T,0);
    par([&](int w){ int bk; while((bk=next.fetch_add(1))<NBK){ sort(out.begin()+bstart[bk],out.begin()+bstart[bk+1]); size_t s=bstart[bk],e=bstart[bk+1]; for(size_t i=s;i<e;){ size_t j=i; while(j<e&&out[j]==out[i])++j; if(j-i==1)++nbw[w]; i=j; } } });
    size_t tot=0; for(auto x:nbw) tot+=x;
    printf("bucket+sort %d threads: %.2fs boundary=%zu\n",T,now()-t,tot);
  }
}
