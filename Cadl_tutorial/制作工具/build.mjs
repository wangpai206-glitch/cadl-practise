import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

let marked;
try { ({ marked } = await import('marked')); }
catch { ({ marked } = await import('/Users/wangpaismacbook/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/marked/lib/marked.esm.js')); }
const directory=path.dirname(fileURLToPath(import.meta.url));
const root=path.dirname(directory);
let html=marked.parse(fs.readFileSync(path.join(root,'CADL新手教程.md'),'utf8'),{gfm:true});
const navigation=[];const seen=new Set();
html=html.replace(/<h2>(.*?)<\/h2>/g,(_,title)=>{
  const match=title.match(/^(\d+)\s+/);
  if(!match)throw new Error('A chapter heading needs a stable numeric prefix: '+title);
  const n=Number(match[1]);if(seen.has(n))throw new Error('Duplicate chapter '+n);seen.add(n);
  const close=navigation.length?'</section>':'';
  navigation.push(`<a href="#lesson-${n}"><span>${String(n).padStart(2,'0')}</span>${title.replace(/^\d+\s*/,'')}</a>`);
  return `${close}<section class="lesson" id="lesson-${n}"><h2>${title}</h2>`;
})+'</section>';
html=html.replace(/<h1>CADL 新手教程<\/h1>/,'');
html=html.replace(/<table>/g,'<div class="table-scroll"><table>').replace(/<\/table>/g,'</table></div>');
let template=fs.readFileSync(path.join(directory,'阅读页模板.html'),'utf8');
template=template.replace('{{NAV}}',()=>navigation.join('')).replace('{{COUNT}}',()=>String(navigation.length)).replace('{{CONTENT}}',()=>html);
fs.writeFileSync(path.join(root,'开始阅读.html'),template);
console.log(`Built ${navigation.length} chapters with stable lesson IDs.`);
