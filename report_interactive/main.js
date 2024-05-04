function change_text() {
    document.getElementById("content_training").innerHTML = "TESTING!";
}

function option_show(option) {
    document.getElementById("content_training").hidden = true;
    document.getElementById("content_evaluate").hidden = true;
    document.getElementById("content_analyze").hidden = true;

    document.getElementById(option).hidden = false;
}
