package com.fbma.brisa.controllers;

import java.util.List;

import com.fbma.brisa.model.GroundRequest;
import com.fbma.brisa.services.GroundService;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/grounds")
public class GroundsController {

    @Autowired
    private GroundService service;
    
    @GetMapping
    public List<String> findFlights(@RequestBody GroundRequest req){
        return service.findGround(req.getCity());
    }
}
