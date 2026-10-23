// Vercel serverless API for FairBench
export default async function handler(req, res) {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  const neonUrl = process.env.NEON_DATABASE_URL;
  if (!neonUrl) {
    return res.status(500).json({ error: 'Database not configured' });
  }

  if (req.method === 'POST') {
    return handlePost(req, res, neonUrl);
  }

  if (req.method === 'GET') {
    return handleGet(req, res, neonUrl);
  }

  return res.status(405).json({ error: 'Method not allowed' });
}

async function handlePost(req, res, dbUrl) {
  // Use neon-http to insert
  const { neon } = await import('@neondatabase/serverless');
  const sql = neon(dbUrl);

  try {
    const report = req.body;

    // Validate required fields
    if (!report.model || !report.hardware) {
      return res.status(400).json({ error: 'Missing required fields: model, hardware' });
    }

    // Insert report
    const result = await sql`
      INSERT INTO reports (
        id, timestamp, published, model_name, quantization,
        cpu_model, cpu_cores, cpu_threads, ram_gb,
        gpu_type, gpu_model, gpu_vram_gb, os,
        quality_mmlu, quality_hellaswag, quality_gsm8k,
        speed_ttft_ms, speed_tpot_ms, speed_throughput,
        memory_vram_peak_gb, memory_cpu_ram_peak_gb, memory_model_file_gb
      ) VALUES (
        ${report.id}, ${report.timestamp}, ${report.published || true},
        ${report.model.name}, ${report.model.quantization},
        ${report.hardware.cpu?.model || ''}, ${report.hardware.cpu?.cores || 0},
        ${report.hardware.cpu?.threads || 0}, ${report.hardware.ram_gb || 0},
        ${report.hardware.gpu?.type || 'none'}, ${report.hardware.gpu?.model || ''},
        ${report.hardware.gpu?.vram_gb || 0}, ${report.hardware.os || ''},
        ${report.quality?.mmlu || 0}, ${report.quality?.hellaswag || 0},
        ${report.quality?.gsm8k || 0}, ${report.speed?.ttft_ms || 0},
        ${report.speed?.tpot_ms || 0}, ${report.speed?.throughput || 0},
        ${report.memory?.vram_peak_gb || 0}, ${report.memory?.cpu_ram_peak_gb || 0},
        ${report.memory?.model_file_size_gb || 0}
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
  } catch (error) {
    console.error('Insert error:', error);
    return res.status(500).json({ error: 'Failed to insert report' });
  }
}

async function handleGet(req, res, dbUrl) {
  const { neon } = await import('@neondatabase/serverless');
  const sql = neon(dbUrl);

  try {
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

    const params = [];
    let whereIdx = params.length + 1;

    if (model) {
      query += ` AND model_name ILIKE $${whereIdx}`;
      params.push(`%${model}%`);
    }
    if (gpu) {
      query += ` AND gpu_model ILIKE $${whereIdx}`;
      params.push(`%${gpu}%`);
    }
    if (min_score) {
      const avgScore = `(quality_mmlu + quality_hellaswag + quality_gsm8k) / 3`;
      query += ` AND ${avgScore} >= $${whereIdx}`;
      params.push(parseFloat(min_score));
    }

    query += ` ORDER BY quality_mmlu DESC, speed_throughput DESC LIMIT 100`;

    const result = await sql(query, params);
    return res.status(200).json({ reports: result });
  } catch (error) {
    console.error('Query error:', error);
    return res.status(500).json({ error: 'Failed to fetch reports' });
  }
}
