package com.fbma.brisa.model;

import java.util.Date;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class Flight {
    String source;
    String destiny;
    Date departure_time;
}
