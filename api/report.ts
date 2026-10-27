// POST /api/report - Submit a benchmark report
import type { VercelRequest, VercelResponse } from '@vercel/node';

export const config = {
  api: {
    bodyParser: true,
  },
};

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const neonUrl = process.env.NEON_DATABASE_URL;
  if (!neonUrl) {
    return res.status(500).json({ error: 'Database not configured' });
  }

  try {
    const { neon } = await import('@neondatabase/serverless');
    const sql = neon(neonUrl);
    const report = req.body;

    if (!report || typeof report !== 'object') {
      return res.status(400).json({ error: 'Invalid request body' });
    }

    // Accept both the new format (from reporter.py) and legacy format
    const modelName = report.model_name || report.model?.name;
    const quantization = report.quantization || report.model?.quantization;
    const timestamp = report.timestamp || new Date().toISOString();
    const id = report.run_id || report.id || crypto.randomUUID();
    const published = report.published ?? true;

    if (!modelName || !quantization) {
      return res.status(400).json({ error: 'Missing required fields: model_name, quantization' });
    }

    const hardware = report.hardware || {};
    const quality = report.quality_scores || {};
    const speed = report.speed || {};
    const memory = report.memory || {};

    const result = await sql`
      INSERT INTO reports (
        id, timestamp, published, model_name, quantization,
        cpu_model, cpu_cores, cpu_threads, ram_gb,
        gpu_type, gpu_model, gpu_vram_gb, os,
        quality_mmlu, quality_hellaswag, quality_gsm8k,
        speed_ttft_ms, speed_tpot_ms, speed_throughput,
        memory_vram_peak_gb, memory_cpu_ram_peak_gb, memory_model_file_gb
      ) VALUES (
        ${id}, ${timestamp}, ${published},
        ${modelName}, ${quantization},
        ${hardware.cpu_model || ''}, ${hardware.cpu_cores || 0},
        ${hardware.cpu_threads || 0}, ${hardware.ram_gb || 0},
        ${hardware.gpu_type || 'none'}, ${hardware.gpu_model || ''},
        ${hardware.gpu_vram_gb || 0}, ${hardware.os || ''},
        ${quality.mmlu ?? 0}, ${quality.hellaswag ?? 0}, ${quality.gsm8k ?? 0},
        ${speed.ttft_ms ?? 0}, ${speed.tpot_ms ?? 0}, ${speed.throughput ?? 0},
        ${memory.vram_peak_gb ?? 0}, ${memory.cpu_ram_peak_gb ?? 0},
        ${report.memory_model_file_gb ?? 0}
      )
      RETURNING id, timestamp, published
    `;

    return res.status(201).json({
      success: true,
      report: {
        id: result[0].id,
        timestamp: result[0].timestamp,
        published: result[0].published,
      }
    });
  } catch (error: any) {
    console.error('Insert error:', error);
    return res.status(500).json({ error: 'Failed to insert report', details: error.message });
  }
}
