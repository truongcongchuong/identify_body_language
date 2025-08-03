var record = document.querySelector("#record");
var reset = document.querySelector("#reset");
var pandemic = document.querySelector("#pandemic");
var displayVideo = document.getElementById("video");
var resulft = document.querySelector("#resulft");
var resulft_video = document.querySelector("#canvas");
var RecordCondition = false;

let mediaRecorder;
let recordedChunks = [];
let FileVideo;

pandemic.disabled = true;
async function StartVideo() {
    const stream = await navigator.mediaDevices.getUserMedia({ video:true });
    mediaRecorder = new MediaRecorder(stream);
    displayVideo.srcObject = stream;

    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            recordedChunks.push(event.data)
        }
    }
    mediaRecorder.onstop = () => {
        const blob = new Blob(recordedChunks, { type: "video/webm"});
        FileVideo = new File([blob], "recorder_video.webm", {type: "video/webm"});
        const urlRecordVideo = URL.createObjectURL(blob);
        const tracks = displayVideo.srcObject.getTracks();
        tracks.forEach(track => track.stop());
        displayVideo.srcObject = null;
        displayVideo.src = urlRecordVideo;
        pandemic.disabled = false;
    }
    mediaRecorder.start();
}
function stop() {
    mediaRecorder.stop();
};
function Reset() {
    try {
        recordedChunks = [];
        const urlFileVideo = URL.createObjectURL(FileVideo);
        URL.revokeObjectURL(urlFileVideo);
        pandemic.disabled = true;
    }
    catch (e) {
        console.log(error);
    }
};
function Pandemic() {
    const formData = new FormData();
    formData.append("video", FileVideo);
    fetch("/frame", {
        method: "POST",
        body: formData,
    })
    .then(response => response.text())
    .then(data => {resulft.innerHTML = data})
    .catch(error => {
        consle.log("lỗi gửi yêu cầu: ", error);
    })
};
function Record() {
    let icon = document.querySelector("#icon-record");
    if (RecordCondition == false) {
        StartVideo();
        icon.className = "bx bx-pause"
        RecordCondition = true;
    }
    else if (RecordCondition == true) {
        stop();
        RecordCondition = false;
        icon.className = "bx bx-play"
    }
};
record.addEventListener("click", Record)
pandemic.addEventListener("click", Pandemic)
reset.addEventListener("click", Reset)