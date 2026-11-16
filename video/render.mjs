// Renderiza index.html cuadro por cuadro con Chrome headless (CDP) y lo entrega a ffmpeg.
// Uso: node render.mjs <page.html> stills <t1,t2,...> <dir>
//      node render.mjs <page.html> video <out.mp4> <fps> [desde] [hasta]
import { spawn } from "node:child_process";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const [page, mode, a, b, c, d] = process.argv.slice(2);
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const port = 9300 + Math.floor(Math.random() * 500);
const chrome = spawn(CHROME, ["--headless=new", `--remote-debugging-port=${port}`, "--window-size=1920,1080",
  "--hide-scrollbars", "--force-device-scale-factor=1", "--mute-audio", `--user-data-dir=${mkdtempSync(join(tmpdir(), "rmx-"))}`,
  "about:blank"], { stdio: "ignore" });

let ws, id = 0; const pending = new Map();
for (let i = 0; i < 50 && !ws; i++) {
  try {
    const list = await (await fetch(`http://127.0.0.1:${port}/json`)).json();
    const tab = list.find(x => x.type === "page");
    ws = new WebSocket(tab.webSocketDebuggerUrl);
    await new Promise((ok, ko) => { ws.onopen = ok; ws.onerror = ko; });
  } catch { ws = null; await new Promise(r => setTimeout(r, 200)); }
}
if (!ws) { chrome.kill(); throw new Error("no se pudo conectar con Chrome"); }
ws.onmessage = e => { const m = JSON.parse(e.data); if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); } };
const send = (method, params = {}) => new Promise((ok, ko) => {
  const i = ++id; pending.set(i, m => m.error ? ko(new Error(method + ": " + m.error.message)) : ok(m.result));
  ws.send(JSON.stringify({ id: i, method, params }));
});
const evalJS = async expr => {
  const r = await send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true });
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text);
  return r.result.value;
};

await send("Page.enable");
await send("Emulation.setDeviceMetricsOverride", { width: 1920, height: 1080, deviceScaleFactor: 1, mobile: false });
await send("Page.navigate", { url: "file://" + resolve(page) });
for (let i = 0; i < 100 && !(await evalJS("typeof window.seek==='function' && document.readyState==='complete'").catch(() => false)); i++)
  await new Promise(r => setTimeout(r, 100));
await evalJS("document.fonts.ready.then(()=>1)");
const total = await evalJS("TL.total");

const shot = async (t, format) => {
  await evalJS(`seek(${t})`);
  const { data } = await send("Page.captureScreenshot", { format, quality: format === "jpeg" ? 94 : undefined });
  return Buffer.from(data, "base64");
};

if (mode === "stills") {
  for (const t of a.split(",").map(Number)) writeFileSync(join(b, `t${String(t).padStart(6, "0")}.png`), await shot(t, "png"));
} else {
  const fps = +b, from = +(c || 0), to = +(d || total), n = Math.round((to - from) * fps);
  const ff = spawn("ffmpeg", ["-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", String(fps), "-c:v", "mjpeg", "-i", "-",
    "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart", a], { stdio: ["pipe", "inherit", "inherit"] });
  const t0 = Date.now();
  for (let f = 0; f < n; f++) {
    const buf = await shot(from + f / fps, "jpeg");
    if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once("drain", r));
    if (f % 150 === 0) console.log(`cuadro ${f}/${n}  ${((Date.now() - t0) / 1000).toFixed(0)}s`);
  }
  ff.stdin.end();
  await new Promise(r => ff.on("close", r));
}
ws.close(); chrome.kill();
