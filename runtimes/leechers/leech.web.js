// leech.web.js
// A simple Node.js helper to perform web searches via DuckDuckGo Instant Answer API.
// Supports both single query mode and batch mode using queries.web.json

// Usage (from command line):
//   node leech.web.js "search query"           # Single query mode
//   node leech.web.js --batch                  # Batch mode using queries.web.json
//   node leech.web.js --batch --output results.json

const fs = require('fs');
const path = require('path');

// Fetch implementation
const fetch = (...args) => import('node-fetch').then(({default: fetch}) => fetch(...args));

async function ddgSearch(query) {
  const params = new URLSearchParams({q: query, format: 'json', no_html: '1', no_redirect: '1'});
  const url = `https://api.duckduckgo.com/?${params.toString()}`;

  try {
    const res = await fetch(url, { headers: { 'Accept': 'application/json' } });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    // Extract plaintext results and related topics
    const results = [];
    if (data.AbstractText) {
      results.push({source: 'abstract', text: data.AbstractText, query: query});
    }
    if (Array.isArray(data.RelatedTopics)) {
      for (const t of data.RelatedTopics) {
        if (t.Text) results.push({source: 'related', text: t.Text, query: query});
      }
    }

    return results;
  } catch (err) {
    console.error(`Error searching "${query}":`, err.message);
    return [];
  }
}

async function loadQueries(queriesFile) {
  const queriesPath = path.resolve(queriesFile || path.join(__dirname, 'queries.web.json'));

  if (!fs.existsSync(queriesPath)) {
    console.error(`Queries file not found: ${queriesPath}`);
    process.exit(1);
  }

  try {
    const content = fs.readFileSync(queriesPath, 'utf-8');
    const config = JSON.parse(content);

    if (!Array.isArray(config.queries)) {
      console.error('Invalid queries.web.json: "queries" must be an array');
      process.exit(1);
    }

    return {
      queries: config.queries,
      settings: config._settings || {}
    };
  } catch (err) {
    console.error(`Error loading queries file: ${err.message}`);
    process.exit(1);
  }
}

async function runBatch(outputFile) {
  const { queries, settings } = await loadQueries();
  const maxResults = settings.max_results_per_query || 10;
  const delay = settings.delay_between_queries_ms || 1000;

  console.log(`Starting batch search with ${queries.length} queries...`);

  const allResults = [];

  for (let i = 0; i < queries.length; i++) {
    const query = queries[i];
    console.log(`[${i + 1}/${queries.length}] Searching: ${query}`);

    const results = await ddgSearch(query);
    const limitedResults = results.slice(0, maxResults);
    allResults.push(...limitedResults);

    if (i < queries.length - 1 && delay > 0) {
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }

  const output = {
    timestamp: new Date().toISOString(),
    total_queries: queries.length,
    total_results: allResults.length,
    results: allResults
  };

  if (outputFile) {
    fs.writeFileSync(outputFile, JSON.stringify(output, null, 2), 'utf-8');
    console.log(`\nResults saved to: ${outputFile}`);
  } else {
    console.log(JSON.stringify(output, null, 2));
  }

  return output;
}

// CLI entrypoint
if (require.main === module) {
  (async () => {
    const args = process.argv.slice(2);

    // Check for batch mode
    if (args.includes('--batch') || args.includes('-b')) {
      const outputIndex = args.indexOf('--output');
      const outputFile = outputIndex !== -1 ? args[outputIndex + 1] : null;
      await runBatch(outputFile);
      return;
    }

    // Single query mode
    const q = args.filter(a => !a.startsWith('-')).join(' ').trim();
    if (!q) {
      console.error('Usage:');
      console.error('  node leech.web.js "search query"     # Single query mode');
      console.error('  node leech.web.js --batch            # Batch mode using queries.web.json');
      console.error('  node leech.web.js --batch --output results.json');
      process.exit(2);
    }

    try {
      const out = await ddgSearch(q);
      console.log(JSON.stringify(out, null, 2));
    } catch (err) {
      console.error('Error:', err.message);
      process.exit(1);
    }
  })();
}

module.exports = { ddgSearch, runBatch, loadQueries };