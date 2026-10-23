// GET /api/models - Get model statistics (averages)
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

    // Get all models with stats
    const result = await sql`
      SELECT 
        model_name,
        ROUND(AVG(quality_mmlu)::numeric, 4) as avg_mmlu,
        ROUND(AVG(quality_hellaswag)::numeric, 4) as avg_hellaswag,
        ROUND(AVG(quality_gsm8k)::numeric, 4) as avg_gsm8k,
        ROUND(AVG(speed_throughput)::numeric, 1) as avg_throughput,
        ROUND(AVG(memory_vram_peak_gb)::numeric, 1) as avg_vram_gb,
        COUNT(*) as sample_count
      FROM reports
      WHERE published = true
      GROUP BY model_name
      ORDER BY avg_mmlu DESC
    `;

    return res.status(200).json({ models: result });
  } catch (error: any) {
    console.error('Query error:', error);
    return res.status(500).json({ error: 'Failed to fetch model stats', details: error.message });
  }
}
