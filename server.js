const http = require("http");

const server = http.createServer((request, response) => {
    const { rawHeaders, httpVersion, method, socket, url } = request;
    const { remoteAddress, remoteFamily } = socket;
    console.log(request);
    // console.log(
    //   JSON.stringify({
    //     rawHeaders,
    //     httpVersion,
    //     method,
    //     remoteAddress,
    //     remoteFamily,
    //     url
    //   })
    // );
  
    response.end();
  });

server.listen(8888);