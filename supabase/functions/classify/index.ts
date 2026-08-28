// Supabase Edge Function - HF Inference API proxy (free, no 50M limit, no private)
// Hosts your 344M ONNX (model.onnx + model.onnx.data) via HF Hub Inference - HF handles ONNX
// Deploy: npx supabase link --project-ref xjjlsxsdjbgrhsnncfdg && npx supabase functions deploy classify --no-verify-jwt
// Env: HF_TOKEN (hf_vip...), SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY

import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.0";

const HF_MODEL = Deno.env.get("HF_MODEL") ?? "Joe-3/smart-recycle-bin";
const HF_TOKEN = Deno.env.get("HF_TOKEN") ?? Deno.env.get("HUGGINGFACE_TOKEN") ?? "";
const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;

const BIN_MAP: Record<string, string> = {
  // 5-class fine-tuned (your category_augmented) 99.38% - 344M ONNX
  biological: "organic", metal: "metal", paper: "paper", plastic: "plastic", trash: "general",
  // 12-class base fallback (watersplash)
  Battery: "hazardous", Biological: "organic", "Brown-glass": "glass",
  Cardboard: "paper", Clothes: "textile", "Green-Glass": "glass",
  Metal: "metal", Paper: "paper", Plastic: "plastic", Shoes: "textile",
  Trash: "general", "White-Glass": "glass",
};

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders() });
  }
  const url = new URL(req.url);
  if (url.pathname.endsWith("/health")) {
    return json({ status: "healthy", model: HF_MODEL, timestamp: new Date().toISOString() });
  }
  if (req.method !== "POST") return json({ error: "POST image file as multipart/form-data field 'file'" }, 405);

  try {
    const form = await req.formData();
    const file = form.get("file") as File | null;
    if (!file) return json({ error: "Missing 'file' field" }, 400);

    const bytes = new Uint8Array(await file.arrayBuffer());

    // 1. Call HF Inference API via router (HF handles 344M ONNX, Edge Function just proxies - no 50M Storage limit)
    const hfRes = await fetch(`https://router.huggingface.co/hf-inference/models/${HF_MODEL}`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${HF_TOKEN}`,
        "Content-Type": "application/octet-stream",
      },
      body: bytes,
    });
    if (!hfRes.ok) {
      const txt = await hfRes.text();
      return json({ error: `HF inference failed: ${hfRes.status} ${txt}` }, 502);
    }
    const hfData: { label: string; score: number }[] = await hfRes.json();

    const top = hfData[0];
    const predicted = top.label;
    const confidence = top.score;
    const allScores: Record<string, number> = {};
    for (const r of hfData) allScores[r.label] = r.score;

    let savedId: string | null = null;
    try {
      const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);
      const fileName = `${crypto.randomUUID()}_${file.name || "image.jpg"}`;
      const path = `predictions/${fileName}`;
      const { error: upErr } = await supabase.storage.from("images").upload(path, bytes, {
        contentType: file.type || "image/jpeg",
        upsert: false,
      });
      const imagePath = upErr ? `predictions/${fileName}` : path;

      const { data, error } = await supabase.from("predictions").insert({
        image_path: imagePath,
        predicted_label: predicted,
        confidence_score: confidence,
        all_scores: allScores,
      }).select("id").single();
      if (!error && data) savedId = data.id;
    } catch (_e) {}

    return json({
      predictedLabel: predicted,
      confidenceScore: confidence,
      bin: BIN_MAP[predicted] ?? "unknown",
      allScores,
      id: savedId,
      model: HF_MODEL,
    });
  } catch (e) {
    return json({ error: String(e) }, 500);
  }
});

function corsHeaders() {
  return {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
    "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  };
}
function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...corsHeaders() },
  });
}
