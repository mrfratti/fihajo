function change_text() {
    document.getElementById("content_training").innerHTML = "TESTING!";
}

function option_show(option) {
    var all_div = document.querySelectorAll('.display');
    for (var nr = 0; nr < all_div.length; nr++) {
        all_div[nr].classList.remove('display');
        all_div[nr].classList.add('hide');
    }

    document.getElementById(option).classList.remove('hide');
    document.getElementById(option).classList.add('display');
}
