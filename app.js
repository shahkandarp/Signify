const express = require("express");
const axios = require('axios')
const cors = require("cors");
const multer = require("multer");
const {
  ref,
  uploadBytes,
  listAll,
  deleteObject,
} = require("firebase/storage");
const storage = require("./firebase");

const app = express();
app.use(cors());
app.use(express.json());


const memoStorage = multer.memoryStorage();
const upload = multer({ memoStorage });
// const options = {
//   hostname: 'localhost:5000',
//   path: '/abc',
//   method: 'GET',
//   headers: {
//     'Content-Type': 'application/json',
//   },
// };

// const getPosts = () => {
//   let data = '';

//   const request = http.request(options, (response) => {
//     // Set the encoding, so we don't get log to the console a bunch of gibberish binary data
//     response.setEncoding('utf8');

//     // // As data starts streaming in, add each chunk to "data"
//     response.on('data', (chunk) => {
//       data += chunk;
//     });

//     // The whole response has been received. Print out the result.
//     response.on('end', () => {
//       console.log(data);
//     });
//   });

//   // Log errors if any occur
//   request.on('error', (error) => {
//     console.error(error);
//   });

//   // End the request
//   request.end();
// };

app.post("/addPicture", upload.single("pic"), async (req, res) => {
  const file = req.file;
  const imageRef = ref(storage, file.originalname);
  const metatype = { contentType: file.mimetype, name: file.originalname };
  const snapshot = await uploadBytes(imageRef, file.buffer, metatype)
    // .then((snapshot) => {
    //   console.log(snapshot.ref._location.bucket)
    //   console.log(snapshot.ref._location.path_)
    //   res.send("uploaded!");
    // })
    // .catch((error) => console.log(error.message));
    console.log(snapshot.ref._location.bucket)
    console.log(snapshot.ref._location.path_)
    const data = await axios.post(`http://127.0.0.1:5000/`,{
		url:`https://firebasestorage.googleapis.com/v0/b/${snapshot.ref._location.bucket}/o/${snapshot.ref._location.path_}?alt=media`
	})
  console.log(data.data)
  res.json({res:"Success",data:data.data.data})
});

app.get('/',async(req,res)=>{
  const data = await axios.post(`http://127.0.0.1:5000/abc`,{
		title: "Foo",
		body: "bar",
		userID: 1
	})
  console.log(data)
      
  console.log('hello');
  res.json({res:"Success"})
})




const port = process.env.PORT || 3001;

const start = async () => {
    app.listen(port, () =>
      console.log(`Server is listening on port ${port}...`))
    
};

//connecting to database
start();