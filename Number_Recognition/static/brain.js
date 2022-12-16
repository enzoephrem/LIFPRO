window.onload = function(){

	var file_input = document.getElementById('file_input')
	var file_upload = document.getElementById('file_upload')
	var image_output = document.getElementById('image_output')

	file_input.addEventListener('change', (event) => {
		console.log(event.target.files[0])
		document.getElementById('file_label').innerHTML = event.target.files[0].name
	})

	file_upload.addEventListener('click', (event) => {
		console.log(file_input.files[0])
		image_output.src = URL.createObjectURL(file_input.files[0])
	})

}