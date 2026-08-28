/**
 * Model listing and blacklist filtering for the DeepSeek provider (`deepseek-official`).
 *
 * The provider no longer ships a hardcoded catalog: the models shown are the
 * ones the configured endpoint (`DEEPSEEK_BASE_URL`, e.g. `newmodel.h3c.com`)
 * advertises over its OpenAI-compatible `GET /v1/models`, minus a configurable
 * blacklist. This module answers both consumers:
 *   - the adapter's `listModels` (the model selector) and
 *   - the "fetch available models" model-discovery offer the plugin registers.
 *
 * Blacklist rules are plain strings; a trailing `*` marks a prefix rule,
 * anything else is an exact id match. A model is hidden when any rule matches.
 *
 * @module dsh-llm-deepseek/models
 */

import { attributionHeaders, LlmError } from '@deepseek-ai/dsh-llm'
import type { LlmDiscoveredModel } from '@deepseek-ai/dsh-llm'

/** Endpoint replies larger than this are refused rather than partially parsed. */
const MAX_RESPONSE_BYTES = 4 * 1024 * 1024

/** One entry of an OpenAI-compatible `GET /v1/models` reply. */
interface ListingEntry {
  id?: unknown
  /** Common gateway extensions; absent from the official listings. */
  name?: unknown
  display_name?: unknown
  context_window?: unknown
  context_length?: unknown
  max_tokens?: unknown
  max_output_tokens?: unknown
}

/**
 * Default models excluded from the selector. Kept in the provider so the
 * template can ship one package; `DSH_MODEL_BLACKLIST` (comma-separated)
 * overrides it at launch.
 */
export const DEFAULT_MODEL_BLACKLIST: readonly string[] = [
  'aix*',
  'blue-model',
  'comware-model',
  'comware-model-air',
  'os-model',
  'deepseek-v4-flash-openai',
  'claude*',
]

/** Whether one model id is hidden by any blacklist rule. */
export function isModelBlacklisted(id: string, rules: readonly string[]): boolean {
  return rules.some((rule) => {
    if (rule.length === 0) return false
    // A trailing `*` is a prefix rule; anything else is exact.
    if (rule.endsWith('*')) return id.startsWith(rule.slice(0, -1))
    return id === rule
  })
}

/** Drop blank rules and normalise a raw list of pattern strings. */
export function normaliseBlacklist(rules: readonly (string | undefined)[]): string[] {
  return rules.filter((rule): rule is string => typeof rule === 'string' && rule.length > 0)
}

/**
 * Join the endpoint base with the listing path, treating the base as a prefix.
 * A base that already ends in `/v1` is not doubled, so both
 * `http://newmodel.h3c.com` and `http://newmodel.h3c.com/v1` resolve to the
 * canonical OpenAI-compatible `${base}/v1/models`.
 */
function listingUrl(baseURL: string): string {
  const base = baseURL.replace(/\/+$/, '')
  return base.endsWith('/v1') ? `${base}/models` : `${base}/v1/models`
}

function label(...candidates: readonly unknown[]): string | undefined {
  for (const candidate of candidates) {
    if (typeof candidate === 'string' && candidate.length > 0) return candidate
  }
  return undefined
}

function capacity(...candidates: readonly unknown[]): number | undefined {
  for (const candidate of candidates) {
    if (typeof candidate === 'number' && Number.isInteger(candidate) && candidate > 0) return candidate
  }
  return undefined
}

/**
 * Read one OpenAI-compatible listing reply. Entries without a usable id are
 * skipped; blacklisted ids are dropped before being returned.
 */
function readListing(body: unknown, blacklist: readonly string[]): LlmDiscoveredModel[] {
  const data = (body as { data?: unknown } | null)?.data
  if (!Array.isArray(data)) {
    throw new LlmError(
      'the endpoint\'s model listing has no "data" array; enter this provider\'s models by hand',
      'DISCOVERY_FAILED',
    )
  }
  const models: LlmDiscoveredModel[] = []
  for (const raw of data) {
    const entry = raw as ListingEntry | null
    const id = label(entry?.id)
    if (id === undefined) continue
    if (isModelBlacklisted(id, blacklist)) continue
    const name = label(entry?.name, entry?.display_name)
    const contextWindow = capacity(entry?.context_window, entry?.context_length)
    const maxTokens = capacity(entry?.max_output_tokens, entry?.max_tokens)
    models.push({
      id,
      ...name === undefined ? {} : { name },
      ...contextWindow === undefined ? {} : { contextWindow },
      ...maxTokens === undefined ? {} : { maxTokens },
    })
  }
  return models
}

/**
 * Interrogate one DeepSeek/OpenAI-compatible endpoint for its advertised
 * models, filtering the blacklist.
 * @param baseURL - endpoint base (e.g. `http://newmodel.h3c.com`).
 * @param apiKey - optional bearer token; `undefined` probes unauthenticated.
 * @param blacklist - pattern strings; matches are omitted.
 * @param signal - caller cancellation.
 * @returns the advertised models, in endpoint order, minus blacklisted ones.
 * @throws LlmError on network failure, a non-2xx status, or a non-listing reply.
 */
export async function discoverModels(
  baseURL: string,
  apiKey: string | undefined,
  blacklist: readonly string[],
  signal?: AbortSignal,
): Promise<readonly LlmDiscoveredModel[]> {
  const url = listingUrl(baseURL)
  let response: Response
  try {
    response = await fetch(url, {
      method: 'GET',
      headers: {
        accept: 'application/json',
        ...apiKey === undefined || apiKey.length === 0 ? {} : { authorization: `Bearer ${apiKey}` },
        ...attributionHeaders(),
      },
      ...signal === undefined ? {} : { signal },
    })
  } catch (error: unknown) {
    if (signal?.aborted) {
      throw new LlmError('model discovery aborted by caller', 'ABORTED', { cause: error })
    }
    throw new LlmError(`could not reach ${url}`, 'DISCOVERY_FAILED', { cause: error })
  }
  if (!response.ok) {
    throw new LlmError(
      `${url} answered ${response.status}${response.status === 401 || response.status === 403 ? '; check the API key' : ''}`,
      'DISCOVERY_FAILED',
    )
  }
  const text = await response.text()
  let body: unknown
  try {
    body = JSON.parse(text)
  } catch (error: unknown) {
    throw new LlmError(`${url} did not answer with JSON`, 'DISCOVERY_FAILED', { cause: error })
  }
  return readListing(body, blacklist)
}
