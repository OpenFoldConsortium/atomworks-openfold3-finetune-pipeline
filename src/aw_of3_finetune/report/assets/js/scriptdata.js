// Loads data/<dir>/<id>.js, a script calling window[callback](id, data); script tags work from file://, fetch does not.
const waiters = new Map();

export function loadScriptData(callback, dir, id) {
  return new Promise((resolve, reject) => {
    window[callback] ??= (gotId, data) => {
      const key = `${callback}|${gotId}`;
      for (const done of waiters.get(key) ?? []) done(data);
      waiters.delete(key);
    };
    const key = `${callback}|${id}`;
    const script = document.createElement("script");
    script.src = `data/${dir}/${encodeURIComponent(id)}.js`;
    const finish = (data) => {
      script.remove();
      resolve(data);
    };
    const fail = (message) => {
      script.remove();
      waiters.set(key, (waiters.get(key) ?? []).filter((done) => done !== finish));
      reject(new Error(message));
    };
    waiters.set(key, [...(waiters.get(key) ?? []), finish]);
    script.onerror = () => fail(`could not load ${script.src}`);
    // A script runs before its load event, so data it hands back has already resolved the promise.
    script.onload = () => fail(`${script.src} did not provide data`);
    document.head.append(script);
  });
}
