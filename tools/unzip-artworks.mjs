// Unpacks frontend/assets/artworks.zip in place during the Vercel build (the build image has neither unzip nor
// Python). Plain Node: reads the zip's central directory and inflates each entry with zlib.
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { inflateRawSync } from 'node:zlib';
import { fileURLToPath } from 'node:url';

// the zip is found relative to this file, so the build works whatever folder Vercel starts in
const dir = join(dirname(fileURLToPath(import.meta.url)), '..', 'frontend', 'assets');
console.log('reading', join(dir, 'artworks.zip'));
const buf = readFileSync(join(dir, 'artworks.zip'));
let eocd = buf.length - 22;
while (eocd >= 0 && buf.readUInt32LE(eocd) !== 0x06054b50) eocd--;
if (eocd < 0) throw new Error('artworks.zip: end of central directory not found');
let p = buf.readUInt32LE(eocd + 16), n = buf.readUInt16LE(eocd + 10), count = 0;
for (let i = 0; i < n; i++) {
  if (buf.readUInt32LE(p) !== 0x02014b50) throw new Error('artworks.zip: bad central directory entry');
  const method = buf.readUInt16LE(p + 10), csize = buf.readUInt32LE(p + 20);
  const nlen = buf.readUInt16LE(p + 28), elen = buf.readUInt16LE(p + 30), clen = buf.readUInt16LE(p + 32);
  const off = buf.readUInt32LE(p + 42), name = buf.toString('utf8', p + 46, p + 46 + nlen);
  p += 46 + nlen + elen + clen;
  if (name.endsWith('/') || name.includes('..')) continue;
  const lstart = off + 30 + buf.readUInt16LE(off + 26) + buf.readUInt16LE(off + 28);
  const data = buf.subarray(lstart, lstart + csize);
  const out = method === 0 ? data : method === 8 ? inflateRawSync(data) : null;
  if (!out) throw new Error(`artworks.zip: unsupported compression for ${name}`);
  const dest = join(dir, name); mkdirSync(dirname(dest), { recursive: true }); writeFileSync(dest, out); count++;
}
console.log(`unpacked ${count} artwork files into ${dir}/artworks`);
