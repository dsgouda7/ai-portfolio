// MathJax v3 config for MkDocs Material + pymdownx.arithmatex (generic: true).
// arithmatex emits <span class="arithmatex">\(...\)</span> and <div class="arithmatex">\[...\]</div>;
// without this config MathJax ignores them and equations render as raw \( ... \) text.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

// Re-typeset on instant-navigation page swaps (Material's navigation.instant feature)
// so equations also render when you click between pages without a full reload.
document$.subscribe(() => {
  if (window.MathJax && window.MathJax.typesetPromise) {
    window.MathJax.typesetPromise();
  }
});
