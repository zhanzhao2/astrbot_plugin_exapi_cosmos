#!/usr/bin/env node
"use strict";
const ExApi=require('./exapi_lib/exApi').default;const EhParse=require('./exapi_lib/ehParse');const EhFetch=require('./exapi_lib/ehFetch');
function out(v){process.stdout.write(JSON.stringify(v)+'\n');}
function read(){return new Promise(r=>{let d='';process.stdin.setEncoding('utf8');process.stdin.on('data',c=>d+=c);process.stdin.on('end',()=>r(d));});}
function nPage(v,d=1){v=parseInt(v,10);return Number.isInteger(v)&&v>0?v:d;}
async function move(search,p){for(let i=2;i<=p;i++){const n=await search.next();if(n===null)break;}return search;}
function pick(a){if(!Array.isArray(a))return[];return a;}
function dec(s){return String(s||'').replace(/&amp;/g,'&').replace(/&quot;/g,'"').replace(/&#39;/g,"'");}
function parseView(html,v){
  const t=String(html||'');
  let m=t.match(/<img[^>]*id="img"[^>]*src="([^"]+)"/i)||t.match(/<img[^>]*src="([^"]+)"[^>]*id="img"/i)||t.match(/<img[^>]*src="([^"]+)"[^>]*style/i);
  const sample=m?dec(m[1]):'';
  m=t.match(/onclick="return nl\('([^\)]+)'\)/i);
  const nl=m?dec(m[1]):'';
  let origin='';
  m=t.match(/<a href="([^"]+)fullimg([^"]+)"/i);
  if(m){origin=dec(m[1])+'fullimg'+dec(m[2]);}
  else{m=t.match(/prompt\('Copy the URL below\.', '([^']+)'\)/i);if(m)origin=dec(m[1]);}
  if(origin&&nl)origin+=(origin.includes('?')?'&':'?')+'nl='+encodeURIComponent(nl);
  const referer=(Array.isArray(v)&&v.length>1)?('https://exhentai.org/s/'+v[0]+'/'+v[1]):'https://exhentai.org/';
  return {sample,origin,best:(origin||sample),referer,nl};
}
async function fetchSearchPage(api,q,p){let h=await api._EhHtml.getSearch(q,1),c=1;while(c<p){const m=String(h||'').match(/href="([^"]*next=[^"<>]+)"/i);if(!m)break;let u=dec(m[1]).replace(/&amp;/g,'&');if(u.startsWith('/'))u='https://exhentai.org'+u;else if(u.startsWith('//'))u='https:'+u;h=await EhFetch.fetch(u);c++;}const s=new EhParse.EhSearch(h,q,api._EhHtml.getSearch);s.page=c;return s;}

(async()=>{
let input={};
try{const raw=(await read()).trim();input=raw?JSON.parse(raw):{};}catch(e){out({ok:false,error:'输入JSON解析失败: '+e.message});process.exitCode=1;return;}
const action=input.action||'';
if(!action){out({ok:false,error:'缺少action'});process.exitCode=1;return;}
const cookies=input.cookies||{};
const miss=['ipb_member_id','ipb_pass_hash','igneous'].filter(k=>!String(cookies[k]||'').trim());
if(miss.length){out({ok:false,error:'缺少Cookie字段: '+miss.join(', ')});process.exitCode=1;return;}
const api=new ExApi(cookies,input.proxy?String(input.proxy):undefined);
try{
if(action==='index'){
  const page=Math.max(0,parseInt(input.page||0,10)||0);
  const idx=await api.getIndex(page);
  out({ok:true,data:{page:page+1,pages:idx.pages,items:idx.getAll()}});
  return;
}
if(action==='search'){
  const keyword=String(input.keyword||'').trim();
  if(!keyword){out({ok:false,error:'keyword不能为空'});process.exitCode=1;return;}
  const page=nPage(input.page,1);
  const s=await fetchSearchPage(api,keyword,page);
  out({ok:true,data:{page:s.page||1,pages:s.pages,items:s.getAll()}});
  return;
}
if(action==='advanced_search'){
  const cfg=(input.config&&typeof input.config==='object'&&!Array.isArray(input.config))?input.config:{};
  const page=nPage(input.page,1);
  const s=await fetchSearchPage(api,cfg,page);
  out({ok:true,data:{page:s.page||1,pages:s.pages,items:s.getAll()}});
  return;
}
if(action==='resolve_views'){const v=input.views;if(!Array.isArray(v)){out({ok:false,error:'views'});process.exitCode=1;return;}const t=3,r=[];for(let c=0;c<v.length;c+=t){const b=v.slice(c,c+t).map(async x=>{try{return parseView(await api._EhHtml.getViewImg(x),x);}catch(_){const ref=(Array.isArray(x)&&x.length>1)?('https://exhentai.org/s/'+x[0]+'/'+x[1]):'https://exhentai.org/';return {sample:'',origin:'',best:'',referer:ref,nl:''};}});r.push(...(await Promise.all(b)));}out({ok:true,data:{candidates:r}});return;}
if(action==='gallery'){
  const h=input.href;
  if(!Array.isArray(h)||h.length<2){out({ok:false,error:'href应为[gid,token]'});process.exitCode=1;return;}
  const ts=parseInt(input.thumb_size,10);
  const thumbType=(ts===0||ts===1)?ts:1;
  const g=await api.getGalleryInfo([String(h[0]),String(h[1])], thumbType);
  const all=Boolean(input.fetch_all_previews);
  const max=Math.max(1,parseInt(input.max_previews||120,10)||120);
  let thumbs=pick(g.getThumbnails()).filter(Boolean);
  let views=pick(g.getViewHref());
  if(all){
    while((g.page||1)<(g.pages||1)&&thumbs.length<max){
      const n=await g.next();
      if(n===null)break;
      views=views.concat(pick(g.getViewHref()));
      thumbs=thumbs.concat(pick(g.getThumbnails()).filter(Boolean));
    }
  }
  views=views.slice(0,max);
  let previewImages=[];
  if(Boolean(input.resolve_preview_images) && views.length>0){
    previewImages=pick(await api.getImgUrl(views));
  }
  let imageCandidates=[];
  if(Boolean(input.resolve_image_candidates) && views.length>0){
    for(const v of views){
      try{
        const html=await api._EhHtml.getViewImg(v);
        imageCandidates.push(parseView(html,v));
      }catch(_){
        const referer=(Array.isArray(v)&&v.length>1)?('https://exhentai.org/s/'+v[0]+'/'+v[1]):'https://exhentai.org/';
        imageCandidates.push({sample:'',origin:'',best:'',referer,nl:''});
      }
    }
  }
  thumbs=[...new Set(thumbs)].slice(0,max);
  out({ok:true,data:{info:g.getAllInfo(),pages:g.pages,comments:pick(g.getComment()),view_hrefs:views,thumbnails:thumbs,preview_images:previewImages,image_candidates:imageCandidates}});
  return;
}
out({ok:false,error:'不支持的action: '+action});process.exitCode=1;
}catch(e){out({ok:false,error:e&&e.message?e.message:String(e)});process.exitCode=1;}
})();
