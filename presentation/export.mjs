import { chromium } from 'playwright';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import fs from 'node:fs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const htmlPath = path.join(__dirname, 'deck.html');
const pdfPath = path.join(__dirname, 'deck.pdf');

const browser = await chromium.launch();
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
  await page.goto('file://' + htmlPath, { waitUntil: 'networkidle' });

  // Count slides so we can sanity-check the page count after export.
  const slideCount = await page.locator('.slide').count();
  console.log(`Slides found: ${slideCount}`);

  await page.pdf({
    path: pdfPath,
    printBackground: true,
    preferCSSPageSize: true,
    displayHeaderFooter: false,
  });

  const bytes = fs.statSync(pdfPath).size;
  console.log(`Wrote ${pdfPath} (${(bytes / 1024).toFixed(1)} KiB)`);
} finally {
  await browser.close();
}
