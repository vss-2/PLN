package com.fbma.brisa.controllers;

import java.util.List;

import com.fbma.brisa.model.Flight;
import com.fbma.brisa.services.FlightService;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/flights")
public class FlightsController {

    @Autowired
    private FlightService service;
    
    @GetMapping
    public List<Flight> findFlights(@RequestBody Flight flight){
        logger.info(flight.getDeparture_time().getTime()+"");
        return service.findFlights(flight.getSource(), flight.getDestiny(), flight.getDeparture_time());
    }
}
