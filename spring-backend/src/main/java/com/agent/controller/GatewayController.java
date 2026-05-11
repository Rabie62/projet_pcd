package com.agent.controller;

import jakarta.servlet.http.HttpServletRequest;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.HttpStatusCodeException;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.net.URI;
import java.net.URISyntaxException;
import java.util.Enumeration;

@RestController
@CrossOrigin(origins = "*", allowedHeaders = "*")
public class GatewayController {

    @Value("${python.service.url:http://medical-ai-agent:8000}")
    private String pythonServiceUrl;

    @Autowired
    private RestTemplate restTemplate;

    @PostMapping(value = "/api/v1/analyze/upload", consumes = org.springframework.http.MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<byte[]> proxyUpload(
            @RequestParam("file") MultipartFile file,
            @RequestParam(value = "patient_id", required = false) String patientId,
            @RequestParam(value = "clinical_notes", required = false) String clinicalNotes) throws Exception {

        String urlString = pythonServiceUrl + "/api/v1/analyze/upload";
        
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(org.springframework.http.MediaType.MULTIPART_FORM_DATA);
        
        org.springframework.util.MultiValueMap<String, Object> body = new org.springframework.util.LinkedMultiValueMap<>();
        body.add("file", file.getResource());
        if (patientId != null) body.add("patient_id", patientId);
        if (clinicalNotes != null) body.add("clinical_notes", clinicalNotes);

        HttpEntity<org.springframework.util.MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

        return restTemplate.exchange(urlString, HttpMethod.POST, requestEntity, byte[].class);
    }

    @PostMapping(value = "/api/v1/knowledge/upload", consumes = org.springframework.http.MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<byte[]> proxyKnowledgeUpload(
            @RequestParam("file") MultipartFile file,
            @RequestParam(value = "uploaded_by", required = false) String uploadedBy) throws Exception {

        // FastAPI expects uploaded_by as a query parameter, not a form field
        String urlString = pythonServiceUrl + "/api/v1/knowledge/upload";
        if (uploadedBy != null) {
            urlString += "?uploaded_by=" + java.net.URLEncoder.encode(uploadedBy, "UTF-8");
        }

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(org.springframework.http.MediaType.MULTIPART_FORM_DATA);

        org.springframework.util.MultiValueMap<String, Object> body = new org.springframework.util.LinkedMultiValueMap<>();
        body.add("file", file.getResource());

        HttpEntity<org.springframework.util.MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

        try {
            return restTemplate.exchange(urlString, HttpMethod.POST, requestEntity, byte[].class);
        } catch (HttpStatusCodeException e) {
            return ResponseEntity.status(e.getStatusCode())
                    .headers(e.getResponseHeaders())
                    .body(e.getResponseBodyAsByteArray());
        }
    }

    @RequestMapping(value = "/api/v1/**", method = {RequestMethod.GET, RequestMethod.POST, RequestMethod.PUT, RequestMethod.DELETE})
    public ResponseEntity<byte[]> proxyRequest(HttpServletRequest request, @RequestBody(required = false) byte[] body) throws URISyntaxException {
        
        String requestUri = request.getRequestURI();
        String queryString = request.getQueryString();
        String urlString = pythonServiceUrl + requestUri;
        if (queryString != null) {
            urlString += "?" + queryString;
        }

        URI uri = new URI(urlString);

        HttpHeaders headers = new HttpHeaders();
        Enumeration<String> headerNames = request.getHeaderNames();
        while (headerNames.hasMoreElements()) {
            String headerName = headerNames.nextElement();
            if(!headerName.equalsIgnoreCase("host") 
               && !headerName.equalsIgnoreCase("content-length")
               && !headerName.equalsIgnoreCase("transfer-encoding")){
                headers.add(headerName, request.getHeader(headerName));
            }
        }

        HttpEntity<byte[]> httpEntity = new HttpEntity<>(body, headers);

        try {
            return restTemplate.exchange(uri, HttpMethod.valueOf(request.getMethod()), httpEntity, byte[].class);
        } catch (HttpStatusCodeException e) {
            return ResponseEntity.status(e.getStatusCode())
                    .headers(e.getResponseHeaders())
                    .body(e.getResponseBodyAsByteArray());
        }
    }
}
