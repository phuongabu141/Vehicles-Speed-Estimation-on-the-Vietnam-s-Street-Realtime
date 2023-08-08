$(function() {
    var rangePercent = $('[type="range"]').val();

    $('[type="range"]').on('change input', function() {
        rangePercent = $('[type="range"]').val();

        const Http = new XMLHttpRequest();
        Http.open('POST', '/')
        Http.send(rangePercent)

        $('h4').html(rangePercent + '<span></span>');
        $('[type="range"], h4>span').css('filter', 'hue-rotate(-' + rangePercent + 'deg)');
        // $('h4').css({'transform': 'translateX(calc(-50% - 20px)) scale(' + (1+(rangePercent/100)) + ')', 'left': rangePercent+'%'});
        $('h4').css({'transform': 'translateX(-50%) scale(' + (1+(rangePercent/100)) + ')', 'left': rangePercent+'%'});
    });
});