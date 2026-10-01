// Mobile audit via CDP: device metrics + real interaction + overflow scan.
// node train/mobile_audit.js <url> <outPng>
const URL_ = process.argv[2] || "http://localhost:8902/";
const OUT = process.argv[3] || "/tmp/mobile_cdp.png";
const W = 390, H = 844, DPR = 3;

const fs = require("fs");
const os = require("os");
const path = require("path");
const net = require("net");
const { spawn } = require("child_process");
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function freePort() {
  return new Promise((res) => {
    const s = net.createServer();
    s.listen(0, () => { const p = s.address().port; s.close(() => res(p)); });
  });
}

async function evaluate(client, sessionId, expr) {
  const r = await client.send("Runtime.evaluate", { expression: expr, returnByValue: true }, sessionId);
  return r.result && r.result.value !== undefined ? r.result.value : r.result;
}

(async () => {
  const port = await freePort();
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), "cdp-"));
  const chrome = spawn(CHROME, ["--headless=new", `--remote-debugging-port=${port}`,
    `--user-data-dir=${profile}`, "--no-first-run", "--disable-gpu", "about:blank"], { stdio: "ignore" });
  try {
    let version = null;
    for (let i = 0; i < 80; i++) {
      try { version = await (await fetch(`http://127.0.0.1:${port}/json/version`)).json(); break; }
      catch { await sleep(250); }
    }
    if (!version) throw new Error("chrome did not start");
    const client = {
      _id: 0, _pending: new Map(), _logs: [],
      ws: null,
      send(method, params = {}, sessionId) {
        return new Promise((res) => {
          const id = ++this._id;
          this._pending.set(id, res);
          this.ws.send(JSON.stringify({ id, method, params, sessionId }));
        });
      },
    };
    await new Promise((res, rej) => {
      client.ws = new WebSocket(version.webSocketDebuggerUrl);
      client.ws.onopen = res;
      client.ws.onerror = rej;
    });
    client.ws.onmessage = (m) => {
      const msg = JSON.parse(m.data);
      if (msg.id && client._pending.has(msg.id)) {
        const res = client._pending.get(msg.id);
        client._pending.delete(msg.id);
        res(msg.result);
      } else if (msg.method === "Log.entryAdded") client._logs.push(msg.params.entry);
    };
    const { targetId } = await client.send("Target.createTarget", { url: "about:blank" });
    const { sessionId } = await client.send("Target.attachToTarget", { targetId, flatten: true });
    const S = (m, p) => client.send(m, p, sessionId);
    await S("Page.enable"); await S("Runtime.enable"); await S("Log.enable");
    await S("Emulation.setDeviceMetricsOverride", { width: W, height: H, deviceScaleFactor: DPR, mobile: true });
    await S("Emulation.setTouchEmulationEnabled", { enabled: true });
    await S("Page.navigate", { url: URL_ });
    await sleep(2500);

    const audit = await evaluate(client, sessionId, `(() => {
      const d = document.documentElement, vw = d.clientWidth;
      const overflow = [];
      for (const e of document.querySelectorAll("body *")) {
        const cs = getComputedStyle(e);
        if (cs.position === "fixed" || cs.display === "none") continue;
        const b = e.getBoundingClientRect();
        if (!b.width || !b.height) continue;
        if (b.right > vw + 1 || b.left < -1 && b.left > -9990) overflow.push({tag:e.tagName, cls:String(e.className||"").slice(0,40), l:Math.round(b.left), r:Math.round(b.right)});
      }
      const tiny = [];
      for (const e of document.querySelectorAll("a,button,select,input")) {
        const b = e.getBoundingClientRect();
        if (!b.width || !b.height) continue;
        if (b.height < 24) tiny.push({cls:String(e.className||"").slice(0,30), text:(e.textContent||"").trim().slice(0,20), h:Math.round(b.height)});
      }
      return {
        viewport: vw, hOverflow: d.scrollWidth > vw,
        overflow: overflow.slice(0,8),
        selects: document.querySelectorAll("select[data-feature]").length,
        ageSelect: !!document.querySelector("#age option"),
        modelNote: (document.getElementById("model-note")||{}).textContent,
        tiny: tiny.slice(0,8),
      };
    })()`);

    // interact: choose some values, submit
    await evaluate(client, sessionId, `(() => {
      const g = document.getElementById("gender").value = "0"; // female
      const sels = [...document.querySelectorAll("select[data-feature]")];
      const set = (f,v) => { const s = sels.find(x=>x.dataset.feature===f); if (s) s.value = v; };
      set("Polyuria","1"); set("Polydipsia","1"); set("weakness","1");
      document.getElementById("age").value = "45";
      document.getElementById("risk-form").requestSubmit();
      return 1;
    })()`);
    await sleep(400);
    const after = await evaluate(client, sessionId, `(() => {
      const d = document.documentElement, vw = d.clientWidth;
      const over = [];
      for (const e of document.querySelectorAll("#result *")) {
        const b = e.getBoundingClientRect();
        if (b.width && b.right > vw + 1) over.push({tag:e.tagName, r:Math.round(b.right)});
      }
      return {
        resultVisible: !document.getElementById("result").hidden,
        heading: (document.getElementById("result-heading")||{}).textContent,
        pct: (document.getElementById("result-gauge")||{}).textContent,
        factors: (document.getElementById("factor-list")||{}).children.length,
        chatVisible: !document.getElementById("ask-assistant").hidden,
        resultOverflow: over.slice(0,6),
        hOverflow: d.scrollWidth > vw,
      };
    })()`);
    console.log(JSON.stringify({ viewport: W + "x" + H, audit, after, consoleErrors: client._logs.filter(l=>l.level==="error").map(l=>l.text.slice(0,120)) }, null, 1));
    if (OUT) {
      const shot = await S("Page.captureScreenshot", { format: "png" });
      fs.writeFileSync(OUT, Buffer.from(shot.data, "base64"));
    }
  } finally {
    chrome.kill("SIGKILL");
  }
  process.exit(0);
})().catch((e) => { console.error("AUDIT FAIL", e.message); process.exit(1); });
