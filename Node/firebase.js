const { initializeApp } = require("firebase/app");
const { getStorage } = require("firebase/storage");

// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyBkP-_8YuxEd8uoBLGW3ByD9L4KdCyFnQo",
  authDomain: "ml-model-4d1ab.firebaseapp.com",
  databaseURL: "https://ml-model-4d1ab-default-rtdb.firebaseio.com",
  projectId: "ml-model-4d1ab",
  storageBucket: "ml-model-4d1ab.appspot.com",
  messagingSenderId: "619286862289",
  appId: "1:619286862289:web:762fe3df16cf2f6f2be8f0",
  measurementId: "G-8532VYKT4S"
};

const firebaseApp = initializeApp(firebaseConfig);

module.exports = getStorage(firebaseApp);