// Copy-to-clipboard for the BibTeX block. No other behaviour.
document.addEventListener('click', function (e) {
  var b = e.target.closest ? e.target.closest('.copy') : null;
  if (!b) return;
  var pre = document.getElementById(b.getAttribute('data-target'));
  if (!pre) return;
  var text = pre.textContent;
  function done() { b.textContent = 'Copied'; setTimeout(function () { b.textContent = 'Copy'; }, 1600); }
  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(text).then(done, fallback);
  } else { fallback(); }
  function fallback() {
    var t = document.createElement('textarea');
    t.value = text; t.setAttribute('readonly', ''); t.style.position = 'fixed'; t.style.opacity = '0';
    document.body.appendChild(t); t.select();
    try { document.execCommand('copy'); done(); } catch (err) {}
    document.body.removeChild(t);
  }
});
