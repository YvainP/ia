$(document).ready(function() {
    var $tabs = $('#tabs').tabs();

    //get selected tab
    function getSelectedTabIndex() {
        return $tabs.tabs('option', 'active');
    }

    //get tab contents
    beginTab = $("#tabs ul li:eq(" + getSelectedTabIndex() + ")").find("a");

    loadTabFrame($(beginTab).attr("href"),$(beginTab).attr("rel"));

    $("a.tabref").click(function() {
        loadTabFrame($(this).attr("href"),$(this).attr("rel"));
    });

    //tab switching function
    function loadTabFrame(tab, url) {
        if ($(tab).find("iframe").length == 0) {
            var html = [];
            html.push('<div class="tabIframeWrapper">');
            html.push('<div class="openout"><a href="' + url + '"></a></div><iframe class="iframetab" src="' + url + '">Load Failed?</iframe>');
            html.push('</div>');
            $(tab).append(html.join(""));
            $(tab).find("iframe").height($(window).height()-70);
        }
        return false;
    }
});