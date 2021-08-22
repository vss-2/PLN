package com.fbma.brisa.services;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Random;

import com.fbma.brisa.model.Flight;

import org.springframework.stereotype.Service;

@Service
public class FlightService {
    
    Random rnd = new Random();

    public List<Flight> findFlights(String source, String destiny, Date date){
        List<Flight> flights = new ArrayList<>();
        if(source.equals("Recife") || rnd.nextInt(3) < 2){
            int numberOfFlights = rnd.nextInt(10) + 1;
            long offset = 0l;
            while(numberOfFlights-- >= 0 && (offset += rnd.nextLong()%14400000) < 86400000){
                flights.add(new Flight(source, destiny, new Date(date.getTime() + offset)));
            }
        }
        Collections.sort(flights, (a,b)->a.getDeparture_time().before(b.getDeparture_time())?-1:1);
        return flights;
    }
}
