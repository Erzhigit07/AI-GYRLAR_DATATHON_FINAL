/* Browser contract checks with synthetic API responses, never competition data.
 * Run with Node and Playwright installed; optionally set CHROME_PATH.
 * No live server or recognition model is started.
 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');

async function main() {
  const root = path.resolve(__dirname, '../..');
  const html = fs.readFileSync(path.join(root, 'web/forte_service/index.html'), 'utf8');
  const output = path.join(root, 'outputs/forte-ui-tests/browser');
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({
    headless: true,
    ...(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {}),
  });
  try {
    const page = await browser.newPage({ viewport: { width: 1360, height: 920 } });
    page.setDefaultTimeout(5000);
    const runtimeErrors = [];
    page.on('pageerror', error => runtimeErrors.push(error.message));
    page.on('dialog', dialog => dialog.accept());
    const key = 'synthetic-browser-key-only';
    const api = { auth: 'normal', ready: true, upload: 'done', poll: 'done', calls: [], result: null, pending: null, examples: [], exampleMode: 'normal', exampleResult: null, exampleExplanation: null, examplePending: null };
    await page.route('http://forte.test/**', async route => {
      const request = route.request();
      const url = new URL(request.url());
      const endpoint = url.pathname;
      assert.equal(url.search, '');
      if (endpoint === '/') return route.fulfill({ contentType: 'text/html', body: html });
      if (endpoint === '/healthz') return route.fulfill({ json: { status: 'ok', model_ready: api.ready } });
      if (endpoint === '/api/v1/rules') return route.fulfill({ json: { rules: [{ rule_id: 'synthetic', title: 'Синтетическое правило для теста', source: 'Тестовый источник', clause: 'Пример' }] } });
      api.calls.push(endpoint);
      if (endpoint === '/api/v1/auth/check' && api.auth === 'offline') return route.abort();
      const valid = request.headers()['x-api-key'] === key;
      if (!valid || (endpoint.startsWith('/api/v1/jobs/') && api.poll === 'unauthorized')) return route.fulfill({ status: 401, json: { detail: 'invalid_api_key' } });
      if (endpoint === '/api/v1/auth/check') return route.fulfill({ json: api.auth === 'malformed' ? {} : { authenticated: true } });
      if (endpoint === '/api/v1/examples') {
        if (api.exampleMode === 'missing') return route.fulfill({ status: 404, json: { detail: 'not_found' } });
        return route.fulfill({ json: { source: 'human_annotation', model_generated: false, examples: api.examples } });
      }
      if (endpoint.startsWith('/api/v1/examples/')) {
        if (api.exampleMode === 'unauthorized') return route.fulfill({ status: 401, json: { detail: 'invalid_api_key' } });
        const id = endpoint.split('/')[4];
        assert(api.examples.some(example => example.example_id === id));
        if (endpoint.endsWith('/video')) {
          if (api.exampleMode === 'pending') { api.examplePending = route; return; }
          return route.fulfill({ contentType: 'video/quicktime', body: fs.readFileSync(path.join(__dirname, 'fixtures/synthetic-360x640.mov')) });
        }
        const data = endpoint.endsWith('/explanation') ? api.exampleExplanation : api.exampleResult;
        return route.fulfill({ json: { ...data, job_id: 'manual-' + id, example_id: id, model_generated: api.exampleMode === 'malformed' } });
      }
      if (endpoint === '/api/v1/analyze') {
        if (api.upload === 'pending') { api.pending = route; return; }
        return route.fulfill({ status: 202, json: { job_id: 'synthetic-job' } });
      }
      if (endpoint.startsWith('/api/v1/jobs/')) {
        if (api.poll === 'offline') return route.abort();
        return route.fulfill({ json: { job_id: 'synthetic-job', status: 'done', ...api.result } });
      }
      throw new Error('Unexpected request: ' + endpoint);
    });
    const visible = id => page.locator(id).waitFor({ state: 'visible' });
    async function login() {
      await page.locator('#api-key').fill(key);
      await page.locator('#login-submit').click();
      await visible('#workspace-screen');
      await page.waitForFunction(() => document.querySelector('#examples-status').textContent !== 'Загружаем примеры…');
    }
    async function upload() {
      await page.locator('#video-file').setInputFiles({ name: 'synthetic.mov', mimeType: 'video/quicktime', buffer: Buffer.from('Synthetic fixture, not a real video') });
      await page.locator('#submit').click();
    }
    async function checkPlayerRatio(width, height) {
      await page.waitForFunction(([width, height]) => {
        const box = document.querySelector('#video-viewer').getBoundingClientRect();
        return Math.abs(box.width - box.height * width / height) < 1;
      }, [width, height]);
      const size = await page.locator('#video-viewer').evaluate(viewer => {
        const box = viewer.getBoundingClientRect();
        const parent = viewer.parentElement;
        const css = getComputedStyle(parent);
        return { height: box.height, width: box.width, maxWidth: parent.clientWidth - parseFloat(css.paddingLeft) - parseFloat(css.paddingRight), viewportHeight: innerHeight, overflow: document.documentElement.scrollWidth > innerWidth };
      });
      assert(size.height <= Math.min(size.viewportHeight * 0.7, 720) + 1, JSON.stringify(size));
      assert(size.width <= size.maxWidth + 1, JSON.stringify(size));
      assert.equal(size.overflow, false);
    }
    async function check(name, work) {
      await work();
      console.log('PASS ' + name);
    }

    await page.goto('http://forte.test/');
    await check('login gate, invalid key, key visibility', async () => {
      assert(await page.locator('#workspace-screen').isHidden());
      assert.equal(api.calls.length, 0);
      await page.locator('#api-key').fill('incorrect-synthetic');
      await page.locator('#toggle-key').click();
      assert.equal(await page.locator('#api-key').getAttribute('type'), 'text');
      await page.locator('#login-submit').click();
      await visible('#login-error');
      assert.match(await page.locator('#login-error').innerText(), /Ключ доступа не принят/);
      assert.equal(await page.locator('#api-key').getAttribute('type'), 'password');
      assert.equal(await page.locator('#api-key').getAttribute('aria-invalid'), 'true');
      assert(await page.locator('#workspace-screen').isHidden());
    });
    await check('server unavailable and malformed auth cannot unlock UI', async () => {
      for (const mode of ['offline', 'malformed']) {
        api.auth = mode;
        await page.locator('#api-key').fill(key);
        await page.locator('#login-submit').click();
        await visible('#login-error');
        assert(await page.locator('#workspace-screen').isHidden());
        assert(await page.locator('#login-submit').isEnabled());
      }
      api.auth = 'normal';
    });
    await check('valid login, no persistent credentials, reload requires login', async () => {
      await login();
      assert.equal(await page.locator('#api-key').inputValue(), '');
      assert.equal(await page.evaluate(() => localStorage.length + sessionStorage.length), 0);
      assert.equal((await page.context().cookies()).length, 0);
      await page.reload();
      await visible('#login-screen');
      assert(await page.locator('#workspace-screen').isHidden());
    });
    await check('model readiness is independent of access', async () => {
      api.ready = false;
      await page.reload();
      await login();
      await page.waitForFunction(() => document.querySelector('#health').dataset.ready === 'false');
      assert(await page.locator('#submit').isDisabled());
      await page.locator('#logout').click();
      api.ready = true;
    });
    await check('white login at desktop, tablet and 320px', async () => {
      await page.emulateMedia({ colorScheme: 'dark' });
      for (const width of [1360, 768, 390, 320]) {
        await page.setViewportSize({ width, height: 920 });
        const size = await page.evaluate(() => ({ width: innerWidth, scroll: document.documentElement.scrollWidth, bg: getComputedStyle(document.body).backgroundColor }));
        assert(size.scroll <= size.width, JSON.stringify(size));
        assert.equal(size.bg, 'rgb(255, 255, 255)');
      }
      await page.setViewportSize({ width: 1360, height: 920 });
      await page.emulateMedia({ colorScheme: 'light' });
      await page.reload();
      await page.locator('#health[data-ready="true"]').waitFor();
      await page.screenshot({ path: path.join(output, 'login-white.png'), fullPage: true });
    });
    await login();
    await page.screenshot({ path: path.join(output, 'workspace-white.png'), fullPage: true });
    await check('player follows decoded video dimensions, viewport and file changes', async () => {
      const uploadsBefore = api.calls.filter(endpoint => endpoint === '/api/v1/analyze').length;
      for (const [width, height] of [[360, 640], [640, 360], [480, 480]]) {
        // Real, tiny MOV fixtures require decoding metadata; no API upload is made.
        await page.locator('#video-file').setInputFiles(path.join(__dirname, 'fixtures', 'synthetic-' + width + 'x' + height + '.mov'));
        await page.waitForFunction(([width, height]) => {
          const video = document.querySelector('#video-preview');
          return video.readyState >= 1 && video.videoWidth === width && video.videoHeight === height;
        }, [width, height]);
        await page.locator('#video-preview').evaluate(async video => { await video.play(); video.pause(); });
        for (const viewport of [{ width: 1360, height: 920 }, { width: 390, height: 740 }, { width: 320, height: 568 }, { width: 920, height: 390 }]) {
          await page.setViewportSize(viewport);
          await checkPlayerRatio(width, height);
        }
        await page.setViewportSize({ width: 1360, height: 920 });
        if (height > width) await page.screenshot({ path: path.join(output, 'portrait-player.png'), fullPage: true });
      }
      await page.locator('#video-file').setInputFiles([]);
      assert.equal(await page.locator('#video-viewer').getAttribute('data-sized'), null);
      assert(await page.locator('#video-empty').isVisible());
      assert.equal(api.calls.filter(endpoint => endpoint === '/api/v1/analyze').length, uploadsBefore);
    });
    const jpeg = await page.evaluate(() => {
      const canvas = document.createElement('canvas');
      canvas.width = 480; canvas.height = 270;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#f4f0f2'; ctx.fillRect(0, 0, 480, 270);
      ctx.fillStyle = '#87234d'; ctx.font = '18px sans-serif';
      ctx.fillText('СИНТЕТИЧЕСКИЙ КАДР ДЛЯ ТЕСТА', 60, 140);
      return canvas.toDataURL('image/jpeg').split(',')[1];
    });
    const events = [0.25, 0.75].map((time, index) => ({ type: 'synthetic', timecode_sec: time, description: index ? 'Второе синтетическое событие' : '<img src=x onerror=alert(1)> — текст для проверки безопасного вывода', evidence_frames: [{ timestamp_sec: time, image_base64: jpeg }] }));
    api.result = { verdict: 'violation', violations: events };
    await check('upload, results and Explainable AI with safe text', async () => {
      await upload();
      await visible('#result-body');
      assert.equal(await page.locator('.event-button').count(), 2);
      assert.equal(await page.locator('.xai-button:visible').count(), 1);
      assert(await page.locator('.upload-panel #verdict').isVisible());
      assert.equal(await page.locator('#results-section .frame').count(), 0);
      await page.setViewportSize({ width: 1024, height: 900 });
      const layout = await page.evaluate(() => {
        const sidebar = document.querySelector('.upload-panel').getBoundingClientRect();
        const results = document.querySelector('.results-panel').getBoundingClientRect();
        return { sidebarRight: sidebar.right, resultsLeft: results.left, sidebarTop: sidebar.top, resultsTop: results.top };
      });
      assert(layout.sidebarRight < layout.resultsLeft);
      assert.equal(layout.sidebarTop, layout.resultsTop);
      await visible('#evidence-preview');
      await checkPlayerRatio(480, 270);
      await page.screenshot({ path: path.join(output, 'calm-forte-result.png'), fullPage: true });
      await page.locator('.event-button').last().click();
      assert.equal(await page.locator('.event-button').last().getAttribute('aria-pressed'), 'true');
      await page.locator('#open-explanation').click();
      assert.equal(await page.locator('#xai-event').inputValue(), '1');
      await page.locator('#xai-event').selectOption('0');
      await visible('#xai-dialog');
      assert.equal(await page.locator('#xai-description').innerText(), events[0].description);
      assert.equal(await page.locator('#xai-description img').count(), 0);
      assert.equal(await page.locator('#xai-frames img').count(), 1);
      await page.locator('#xai-source-details summary').click();
      assert.match(await page.locator('#xai-source').innerText(), /Тестовый источник/);
      await page.locator('#xai-event').selectOption('1');
      assert.equal(await page.locator('#xai-description').innerText(), events[1].description);
      await page.locator('#xai-close').focus();
      await page.keyboard.press('Shift+Tab');
      assert.equal(await page.evaluate(() => document.activeElement.id), 'xai-jump');
      await page.keyboard.press('Escape');
      assert(await page.locator('#xai-dialog').isHidden());
      assert(await page.locator('#open-explanation').evaluate(element => element === document.activeElement));
      await page.locator('#open-explanation').click();
      await page.screenshot({ path: path.join(output, 'explainable-ai-white.png') });
      for (const width of [768, 390, 320]) {
        await page.setViewportSize({ width, height: 920 });
        assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
        assert(await page.locator('#xai-dialog').evaluate(dialog => dialog.scrollWidth <= dialog.clientWidth));
      }
      await page.locator('#xai-jump').click();
      assert(await page.locator('#xai-dialog').isHidden());
      await page.setViewportSize({ width: 1360, height: 920 });
      await page.locator('#logout').click();
      assert.equal(await page.locator('#events').textContent(), '');
      assert.equal(await page.locator('#xai-frames img').count(), 0);
      assert.equal(await page.locator('#video-preview').getAttribute('src'), null);
      assert.equal(await page.locator('#file-name').textContent(), 'Файл не выбран');
      assert.equal(await page.locator('#selected-frame').getAttribute('src'), null);
      assert.equal(await page.locator('#video-viewer').getAttribute('data-sized'), null);
    });
    await check('poll retry does not upload twice; 401 clears session', async () => {
      await login();
      api.poll = 'offline';
      await upload();
      await visible('#retry-status');
      const count = api.calls.filter(p => p === '/api/v1/analyze').length;
      api.poll = 'unauthorized';
      await page.locator('#retry-status').click();
      await visible('#login-screen');
      assert.match(await page.locator('#login-status').innerText(), /Войдите повторно/);
      assert.equal(api.calls.filter(p => p === '/api/v1/analyze').length, count);
      assert(await page.locator('#workspace-screen').isHidden());
      api.poll = 'done';
    });
    await check('logout during upload ignores late response', async () => {
      await login();
      api.upload = 'pending';
      await upload();
      await page.waitForFunction(() => document.querySelector('#submit').textContent.includes('Передаём'));
      await page.locator('#logout').click();
      const count = api.calls.length;
      if (api.pending) await api.pending.fulfill({ status: 202, json: { job_id: 'late-synthetic-job' } }).catch(() => {});
      await page.waitForTimeout(100);
      assert.equal(api.calls.length, count);
      assert(await page.locator('#workspace-screen').isHidden());
      assert.equal(await page.locator('#api-key').inputValue(), '');
      api.upload = 'done';
    });
    await check('negative result has no invented explanation', async () => {
      await login();
      api.result = { verdict: 'no_violation', violations: [] };
      await upload();
      await visible('#verdict');
      assert(await page.locator('#open-explanation').isHidden());
      assert.match(await page.locator('#verdict-title').innerText(), /Нарушения не обнаружены/);
      await page.locator('#logout').click();
    });
    api.examples = ['Сейф', 'Дверь', 'Верхняя одежда', 'Постороннее занятие'].map((title, index) => ({ example_id: 'synthetic-' + index, video_id: 'TEST-' + index, title: title + ' — синтетический пример', timecode_sec: 0.25 }));
    api.exampleResult = { status: 'done', result_source: 'human_annotation', verdict: 'violation', violations: [{ ...events[0], type: 'human-only-rule', description: 'Синтетическая ручная отметка. <script> — обычный текст.' }] };
    api.exampleExplanation = { status: 'done', result_source: 'human_annotation', explanation: { aggregation: { method: 'manual_annotation', explanation: 'Пример размечен человеком, модель не запускалась.' }, events: [{ event_index: 0, type: 'human-only-rule', timecode_sec: 0.25, description_source: 'human_annotation', context_frames: ['before', 'event', 'after'].map((role, i) => ({ role, timestamp_sec: [0.1, 0.25, 0.4][i], image_base64: jpeg })), missing_context_roles: [] }] } };
    await check('manual examples work without model, preserve provenance and real context timestamps', async () => {
      api.ready = false;
      await page.reload();
      await login();
      const count = api.calls.filter(endpoint => endpoint === '/api/v1/analyze').length;
      assert.equal(await page.locator('.example-button').count(), 4);
      await page.locator('.example-button').first().click();
      await visible('#result-source');
      await page.waitForFunction(() => Math.abs(document.querySelector('#video-preview').currentTime - 0.25) < 0.02);
      assert.equal(await page.locator('#result-source').innerText(), 'Демонстрационный пример');
      assert.match(await page.locator('.event-copy strong').innerText(), /Сейф/);
      assert(await page.locator('#submit').isHidden());
      await page.locator('#open-explanation').click();
      assert.equal(await page.locator('#xai-provenance').innerText(), 'Демонстрационный пример');
      assert.equal(await page.locator('#xai-description-title').innerText(), 'Объяснение события');
      assert(await page.locator('#xai-confidence').isHidden());
      assert.equal(await page.locator('#xai-frames img').count(), 3);
      assert.deepEqual(await page.locator('#xai-frames figcaption').allTextContents(), ['До события · 00:00.1', 'Момент события · 00:00.2', 'После события · 00:00.4']);
      assert.equal(await page.locator('#xai-description script').count(), 0);
      assert.equal(await page.locator('#xai-source-details summary').innerText(), 'Источник результата');
      await page.locator('#xai-source-details summary').click();
      assert.match(await page.locator('#xai-source').innerText(), /Ручная разметка пользователя/);
      assert.match(await page.locator('#xai-source').innerText(), /модель не принимала это решение/);
      await page.screenshot({ path: path.join(output, 'human-annotation-explanation.png') });
      await page.setViewportSize({ width: 320, height: 700 });
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      assert(await page.locator('#xai-dialog').evaluate(dialog => dialog.scrollWidth <= dialog.clientWidth));
      await page.locator('#xai-frames button').last().click();
      assert(await page.locator('#xai-dialog').isHidden());
      await page.waitForFunction(() => Math.abs(document.querySelector('#video-preview').currentTime - 0.4) < 0.02);
      await page.setViewportSize({ width: 1360, height: 920 });
      await page.locator('.example-button').last().click();
      await page.waitForFunction(() => document.querySelector('#file-name').textContent === 'TEST-3.mov');
      assert.match(await page.locator('.event-copy strong').innerText(), /Постороннее занятие/);
      assert.equal(api.calls.filter(endpoint => endpoint === '/api/v1/analyze').length, count);
      await page.locator('#video-file').setInputFiles(path.join(__dirname, 'fixtures/synthetic-640x360.mov'));
      assert(await page.locator('#result-source').isHidden());
      assert(await page.locator('#verdict').isHidden());
      assert.equal(await page.locator('#xai-frames img').count(), 0);
    });
    await check('manual example rejects wrong provenance and ignores a late response after logout', async () => {
      api.exampleMode = 'malformed';
      await page.locator('.example-button').first().click();
      await page.waitForFunction(() => document.querySelector('#examples-status').textContent.includes('Пример не открыт'));
      assert(await page.locator('#result-source').isHidden());
      assert(await page.locator('.example-button').first().isEnabled());
      api.exampleMode = 'pending';
      const videoRequest = page.waitForRequest(request => request.url().endsWith('/synthetic-0/video'));
      await page.locator('.example-button').first().click();
      await videoRequest;
      await page.locator('#logout').click();
      if (api.examplePending) await api.examplePending.fulfill({ contentType: 'video/quicktime', body: fs.readFileSync(path.join(__dirname, 'fixtures/synthetic-360x640.mov')) }).catch(() => {});
      await page.waitForTimeout(100);
      assert(await page.locator('#workspace-screen').isHidden());
      assert.equal(await page.locator('#video-preview').getAttribute('src'), null);
      assert.equal(await page.locator('.example-button').count(), 0);
      api.exampleMode = 'normal';
    });
    await check('missing example endpoint is recoverable; unauthorized example clears session', async () => {
      api.exampleMode = 'missing';
      await login();
      await visible('#examples-retry');
      assert.match(await page.locator('#examples-status').innerText(), /пока не подключены/);
      api.exampleMode = 'normal';
      await page.locator('#examples-retry').click();
      await page.locator('.example-button').first().waitFor({ state: 'visible' });
      api.exampleMode = 'unauthorized';
      await page.locator('.example-button').first().click();
      await visible('#login-screen');
      assert.match(await page.locator('#login-status').innerText(), /Войдите повторно/);
      assert.equal(await page.locator('.example-button').count(), 0);
    });
    await check('ordinary analysis retains model provenance after viewing human examples', async () => {
      api.ready = true;
      api.exampleMode = 'normal';
      api.result = { verdict: 'violation', violations: events };
      await page.reload();
      await login();
      await upload();
      await visible('#result-body');
      assert(await page.locator('#result-source').isHidden());
      await page.locator('#open-explanation').click();
      assert.equal(await page.locator('#xai-description-title').innerText(), 'Описание модели');
      assert(await page.locator('#xai-provenance').isHidden());
      assert(await page.locator('#xai-confidence').isVisible());
      assert.equal(await page.locator('#xai-frames img').count(), 1);
      await page.locator('#xai-close').click();
      await page.locator('#logout').click();
    });
    assert.deepEqual(runtimeErrors, []);
    console.log('PASS no JavaScript runtime errors');
  } finally {
    await browser.close();
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
