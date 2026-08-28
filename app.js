const copyButton = document.querySelector("#copy-citation");
const citation = document.querySelector("#bibtex code");

if (copyButton && citation) {
  copyButton.addEventListener("click", async () => {
    const originalLabel = copyButton.textContent;
    try {
      await navigator.clipboard.writeText(citation.textContent);
      copyButton.textContent = "Copied";
    } catch (error) {
      const range = document.createRange();
      range.selectNodeContents(citation);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      copyButton.textContent = "Selected";
    }
    window.setTimeout(() => {
      copyButton.textContent = originalLabel;
    }, 1600);
  });
}
