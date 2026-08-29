const lightbox = document.querySelector("#lightbox");
const lightboxImage = document.querySelector("#lightbox-image");
const figureButtons = document.querySelectorAll("[data-lightbox-src]");
const closeButtons = document.querySelectorAll("[data-lightbox-close]");
let lastTrigger = null;

function openLightbox(button) {
  if (!lightbox || !lightboxImage) return;
  lastTrigger = button;
  lightboxImage.src = button.dataset.lightboxSrc;
  lightboxImage.alt = button.dataset.lightboxAlt || "QDBO framework";
  lightbox.hidden = false;
  document.body.classList.add("lightbox-open");
  const closeButton = lightbox.querySelector(".lightbox-close");
  if (closeButton) closeButton.focus();
}

function closeLightbox() {
  if (!lightbox || !lightboxImage) return;
  lightbox.hidden = true;
  lightboxImage.src = "";
  document.body.classList.remove("lightbox-open");
  if (lastTrigger) lastTrigger.focus();
}

figureButtons.forEach((button) => {
  button.addEventListener("click", () => openLightbox(button));
});

closeButtons.forEach((button) => {
  button.addEventListener("click", closeLightbox);
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && lightbox && !lightbox.hidden) closeLightbox();
});
