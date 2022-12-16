window.onload = function () {

	console.log("Test")

	// Definitions
	var canvas = document.getElementById("paint-canvas");
	var context = canvas.getContext("2d");
	var boundings = canvas.getBoundingClientRect();

	// Specifications
	var mouseX = 0;
	var mouseY = 0;
	context.strokeStyle = 'white'; // initial brush color
	context.fillStyle = "black";
	context.fillRect(0, 0, canvas.width, canvas.height);
	context.lineWidth = 25; // initial brush width
	var isDrawing = false;
  
  
	// Handle Colors
	var colors = document.getElementsByClassName('colors')[0];
  
	colors.addEventListener('click', function(event) {
	  context.strokeStyle = event.target.value || 'black';
	});
  
	// Handle Brushes
	var brushes = document.getElementsByClassName('brushes')[0];
  
	brushes.addEventListener('click', function(event) {
	  context.lineWidth = event.target.value || 1;
	});
  
	// Mouse Down Event
	canvas.addEventListener('mousedown', function(event) {
	  setMouseCoordinates(event);
	  isDrawing = true;
  
	  // Start Drawing
	  context.beginPath();
	  context.moveTo(mouseX, mouseY);
	});
  
	// Mouse Move Event
	canvas.addEventListener('mousemove', function(event) {
	  setMouseCoordinates(event);
  
	  if(isDrawing){
		context.lineTo(mouseX, mouseY);
		context.stroke();
	  }
	});
  
	// Mouse Up Event
	canvas.addEventListener('mouseup', function(event) {
	  setMouseCoordinates(event);
	  isDrawing = false;
	});
  
	// Handle Mouse Coordinates
	function setMouseCoordinates(event) {
	  mouseX = event.clientX - boundings.left;
	  mouseY = event.clientY - boundings.top;
	}
  
	// Handle Clear Button
	var clearButton = document.getElementById('clear');
  
	clearButton.addEventListener('click', function() {
	  context.clearRect(0, 0, canvas.width, canvas.height);
	  context.fillStyle = "black";
	  context.fillRect(0, 0, canvas.width, canvas.height);
	});
  
	// Handle Save Button
	var saveButton = document.getElementById('save');
  
	saveButton.addEventListener('click', function() {
	  var imageData = context.getImageData(0, 0, canvas.width, canvas.height);
	  var dataURL = canvas.toDataURL()
		// Use blob to send the image
		$.ajax({
			type: "POST",
			url: "/numbers",
			data: { 
			   image: dataURL
			}
		  }).done(function(o) {
			console.log('saved'); 
			// If you want the file to be visible in the browser 
			// - please modify the callback in javascript. All you
			// need is to return the url to the file, you just saved 
			// and than put the image in your browser.
		  });

  });
};
  

  