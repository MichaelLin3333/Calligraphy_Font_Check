const fileInput = document.getElementById('fileInput');
const useCamera = document.getElementById('useCamera');
const capture = document.getElementById('capture');
const send = document.getElementById('send');
const processedImg = document.getElementById('processed');
const overlayImg = document.getElementById('overlay');
const predictionsDiv = document.getElementById('predictions');
const augmentCheckbox = document.getElementById('augment');

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
let stream = null;
let lastBlob = null;

fileInput.addEventListener('change', (e)=>{
  const f = e.target.files[0];
  if (!f) return;
  lastBlob = f;
  const url = URL.createObjectURL(f);
  processedImg.src = url;
});

useCamera.addEventListener('click', async ()=>{
  if (!stream) {
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = stream;
    video.style.display = 'block';
    capture.disabled = false;
  } else {
    stream.getTracks().forEach(t=>t.stop());
    stream = null;
    video.style.display = 'none';
    capture.disabled = true;
  }
});

capture.addEventListener('click', ()=>{
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0);
  canvas.toBlob((b)=>{
    lastBlob = b;
    const url = URL.createObjectURL(b);
    processedImg.src = url;
  }, 'image/png');
});

send.addEventListener('click', async ()=>{
  if (!lastBlob) { alert('No image selected or captured'); return; }
  const form = new FormData();
  form.append('file', lastBlob, 'upload.png');
  form.append('augment', augmentCheckbox.checked ? '1' : '0');

  const res = await fetch('/visualize', { method: 'POST', body: form });
  if (!res.ok) {
    alert('Server error');
    return;
  }
  const data = await res.json();
  processedImg.src = 'data:image/png;base64,' + data.processed_base64;
  overlayImg.src = 'data:image/png;base64,' + data.saliency_base64;
  let html = `<strong>汉字:</strong> ${data.char_pred} (${(data.char_conf*100).toFixed(2)}%)<br>`;
  html += `<strong>前5个字候选:</strong><br>`;
  data.char_results.forEach(r=>{ html += `${r.char}: ${(r.prob*100).toFixed(2)}%<br>` });
  html += `<strong>书法风格:</strong> ${data.style_pred} (${(data.style_conf*100).toFixed(2)}%)`;
  predictionsDiv.innerHTML = html;
});
