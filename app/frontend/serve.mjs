import { createServer } from "node:http";
import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import { extname, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(fileURLToPath(new URL("./dist", import.meta.url)));
const apiOrigin = process.env.API_ORIGIN || "http://127.0.0.1:8000";
const mimeTypes = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".ico": "image/x-icon",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
};

createServer(async (request, response) => {
  const url = new URL(request.url || "/", "http://localhost");

  if (url.pathname.startsWith("/api/")) {
    try {
      const upstream = await fetch(`${apiOrigin}/${url.pathname.slice(5)}${url.search}`, {
        method: request.method,
        headers: {
          "content-type": request.headers["content-type"] || "application/json",
        },
        body: ["GET", "HEAD"].includes(request.method) ? undefined : request,
        duplex: "half",
      });
      response.writeHead(upstream.status, {
        "content-type": upstream.headers.get("content-type") || "application/json",
      });
      response.end(Buffer.from(await upstream.arrayBuffer()));
    } catch {
      response.writeHead(502, { "content-type": "application/json" });
      response.end(JSON.stringify({ detail: "The prediction API is unavailable." }));
    }
    return;
  }

  let filePath = resolve(root, `.${decodeURIComponent(url.pathname)}`);
  if (filePath !== root && !filePath.startsWith(`${root}${sep}`)) {
    response.writeHead(403).end();
    return;
  }

  try {
    if (!(await stat(filePath)).isFile()) filePath = resolve(root, "index.html");
  } catch {
    filePath = resolve(root, "index.html");
  }

  response.writeHead(200, {
    "content-type": mimeTypes[extname(filePath)] || "application/octet-stream",
  });
  createReadStream(filePath).pipe(response);
}).listen(Number(process.argv[2] || process.env.PORT || 5188), "127.0.0.1");
