package com.fbma.brisa.services;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.springframework.stereotype.Service;
import org.springframework.web.bind.annotation.RequestBody;

@Service
public class GroundService {
    
    Random rnd = new Random();

    public List<String> findGround(@RequestBody String city){
        List<String> grounds = new ArrayList<>();
        grounds.add("Taxi");
        if(city.equals("Recife") || rnd.nextInt(3) < 2){
            if(rnd.nextInt(10) < 8) grounds.add("Buses");
            if(rnd.nextInt(10) < 5) grounds.add("Metro");
            if(rnd.nextInt(10) < 2) grounds.add("Train");
        }
        Collections.sort(grounds);
        return grounds;
    }
}
