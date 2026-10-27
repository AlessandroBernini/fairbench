// Shared database connection helper for Vercel Serverless
import type { VercelRequest, VercelResponse } from '@vercel/node';

let neonPromise: Promise<any> | null = null;

export async function getDb(): Promise<any> {
  if (neonPromise) return neonPromise;
  
  const neonUrl = process.env.NEON_DATABASE_URL;
  if (!neonUrl) {
    throw new Error('Database not configured');
  }
  
  const { neon } = await import('@neondatabase/serverless');
  neonPromise = neon(neonUrl);
  return neonPromise;
}

export function getDbError(): string | null {
  const neonUrl = process.env.NEON_DATABASE_URL;
  if (!neonUrl) return 'Database not configured';
  return null;
}

export function withDbError(
  res: VercelResponse,
  error: string,
  statusCode: number = 500
) {
  return res.status(statusCode).json({ error });
}
