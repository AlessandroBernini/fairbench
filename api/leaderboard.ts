// GET /api/leaderboard - Get leaderboard with optional filters
import type { VercelRequest, VercelResponse } from '@vercel/node';
import { getDb, getDbError } from './db';

export const config = {
  api: {
    bodyParser: true,
  },
  cache: 'no-store',
};

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const dbError = getDbError();
  if (dbError) {
    return res.status(500).json({ error: dbError });
  }

  try {
    const sql = await getDb();

    const { model, gpu, min_score } = req.query;

    const rawLimit = req.query.limit;
    const parsedLimit = parseInt(Array.isArray(rawLimit) ? String(rawLimit[0]) : String(rawLimit ?? "100"), 10);
    const limit = Number.isFinite(parsedLimit) && parsedLimit > 0 ? parsedLimit : 100;

    let query = `
      SELECT 
        id, timestamp, model_name, quantization,
        cpu_model, cpu_cores, ram_gb,
        gpu_type, gpu_model, gpu_vram_gb,
        quality_mmlu, quality_hellaswag, quality_gsm8k,
        speed_ttft_ms, speed_tpot_ms, speed_throughput,
        memory_vram_peak_gb, memory_cpu_ram_peak_gb
      FROM reports
      WHERE published = true
    `;

    const params: any[] = [];
    let idx = 1;

    if (model) {
      query += ` AND model_name ILIKE $${idx}`;
      params.push(`%${model}%`);
      idx++;
    }
    if (gpu) {
      query += ` AND gpu_model ILIKE $${idx}`;
      params.push(`%${gpu}%`);
      idx++;
    }
    if (min_score) {
      const minScore = parseFloat(min_score as string);
      query += ` AND (quality_mmlu + quality_hellaswag + quality_gsm8k) / 3 >= $${idx}`;
      params.push(minScore);
      idx++;
    }

    query += ` ORDER BY quality_mmlu DESC, speed_throughput DESC LIMIT ${limit}`;

    const result = await sql(query, params);
    return res.status(200).json({ reports: result });
  } catch (error: any) {
    console.error('Query error:', error);
    return res.status(500).json({ error: 'Failed to fetch reports', details: error.message });
  }
}
