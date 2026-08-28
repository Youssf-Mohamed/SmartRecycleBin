// Supabase Edge Function - HF Inference API proxy (free, no 50M limit)
// Deployed at https://xjjlsxsdjbgrhsnncfdg.supabase.co/functions/v1/classify
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.0";
const HF_MODEL = Deno.env.get("HF_MODEL") ?? "Joe-3/smart-recycle-bin";
const HF_TOKEN = Deno.env.get("HF_TOKEN") ?? Deno.env.get("HUGGINGFACE_TOKEN") ?? "";
const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const BIN_MAP: Record<string, string> = {
  biological: "organic", metal: "metal", paper: "paper", plastic: "plastic", trash: "general",
  Battery: "hazardous", Biological: "organic", "Brown-glass": "glass", Cardboard: "paper", Clothes: "textile", "Green-Glass": "glass", Metal: "metal", Paper: "paper", Plastic: "plastic", Shoes: "textile", Trash: "general", "White-Glass": "glass",
};
Deno.serve(async (req) => {
  if (req.method === "OPTIONS") return new Response(null, { headers: corsHeaders() });
  const url = new URL(req.url);
  if (url.pathname.endsWith("/health")) return json({ status: "healthy", model: HF_MODEL, timestamp: new Date().toISOString() });
  if (req.method !== "POST") return json({ error: "POST file" }, 405);
  try {
    const form = await req.formData(); const file = form.get("file") as File | null;
    if (!file) return json({ error: "Missing file" }, 400);
    const bytes = new Uint8Array(await file.arrayBuffer());
    const hfRes = await fetch(`https://router.huggingface.co/hf-inference/models/${HF_MODEL}`, {
      method: "POST", headers: { Authorization: `Bearer ${HF_TOKEN}`, "Content-Type": "application/octet-stream" }, body: bytes,
    });
    let hfData: { label: string; score: number }[];
    if (!hfRes.ok) {
      const txt = await hfRes.text();
      if (hfRes.status === 400 && txt.includes("not supported")) {
        return json({ error: `HF Inference not yet deployed for ${HF_MODEL}. Use static Space https://joe-3-smart-recycle-bin-static.hf.space for instant 99.38% inference (loads https://huggingface.co/Joe-3/smart-recycle-bin/resolve/main/model.safetensors via transformers.js, saves to Supabase xjjlsxsdjbgrhsnncfdg).`, hint: "Static Space works now; HF Inference for custom 5-class deploys in 5-60 min or click Ask for provider support at https://huggingface.co/Joe-3/smart-recycle-bin", hf_error: txt }, 503);
      }
      return json({ error: `HF inference failed: ${hfRes.status} ${txt}` }, 502);
    }
    hfData = await hfRes.json();
    const top = hfData[0]; const allScores: Record<string, number> = {}; for (const r of hfData) allScores[r.label] = r.score;
    let savedId: string | null = null;
    try {
      const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);
      const fileName = `${crypto.randomUUID()}_${file.name || "image.jpg"}`; const path = `predictions/${fileName}`;
      const { error: upErr } = await supabase.storage.from("images").upload(path, bytes, { contentType: file.type || "image/jpeg", upsert: false });
      const imagePath = upErr ? `predictions/${fileName}` : path;
      const { data } = await supabase.from("predictions").insert({ image_path: imagePath, predicted_label: top.label, confidence_score: top.score, all_scores: allScores }).select("id").single();
      if (data) savedId = data.id;
    } catch (_e) {}
    return json({ predictedLabel: top.label, confidenceScore: top.score, bin: BIN_MAP[top.label] ?? "unknown", allScores, id: savedId, model: HF_MODEL });
  } catch (e) { return json({ error: String(e) }, 500); }
});
function corsHeaders() { return { "Access-Control-Allow-Origin": "*", "Access-Control-Allow-Methods": "POST, GET, OPTIONS", "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type" }; }
function json(body: unknown, status = 200) { return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json", ...corsHeaders() } }); }
