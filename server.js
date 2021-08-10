// const http = require("http");

// const server = http.createServer((request, response) => {
//     const { rawHeaders, httpVersion, method, socket, url } = request;
//     const { remoteAddress, remoteFamily } = socket;
//     // console.log(request);
//     console.log(
//       JSON.stringify({
//         rawHeaders,
//         httpVersion,
//         method,
//         remoteAddress,
//         remoteFamily,
//         url
//       })
//     );
  
//     response.end();
//   });
// server.listen(8888);

const express = require('express');
const app = express();
const port = 8888

app.post('/airfare_find', (req, res) => {
  // console.log(Object.keys(req));
  console.log(req['socket']);
  res.send('OK! I\'m going to do what you\'ve asked!');
  // res.send('Hello World!');
});

app.listen(port, () => {
  console.log(`Listening at http://localhost:${port}`)
})