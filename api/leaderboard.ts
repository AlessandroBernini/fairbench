// GET /api/leaderboard - Get leaderboard with optional filters
import type { VercelRequest, VercelResponse } from '@vercel/node';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const neonUrl = process.env.NEON_DATABASE_URL;
  if (!neonUrl) {
    return res.status(500).json({ error: 'Database not configured' });
  }

  try {
    const { neon } = await import('@neondatabase/serverless');
    const sql = neon(neonUrl);

    const { model, gpu, min_score } = req.query;
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
      const avgScore = `(quality_mmlu + quality_hellaswag + quality_gsm8k) / 3`;
      query += ` AND ${avgScore} >= $${idx}`;
      params.push(parseFloat(min_score as string));
      idx++;
    }

    query += ` ORDER BY quality_mmlu DESC, speed_throughput DESC LIMIT 100`;

    const result = await sql(query, params);
    return res.status(200).json({ reports: result });
  } catch (error: any) {
    console.error('Query error:', error);
    return res.status(500).json({ error: 'Failed to fetch reports', details: error.message });
  }
}
