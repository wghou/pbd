#pragma once

#include <glm/glm.hpp>
#include <iostream>
#include <chrono>

#define PRINT_VALUE(var) std::cout << #var << " = " << var << std::endl
#define TIMING(expr, event_name) do {Timer t; t.start(); expr; t.stop(); std::cout << event_name << " took " << t.getElapsedMilliseconds() << " ms." << std::endl; } while (0)

typedef std::chrono::system_clock Clock;
typedef std::chrono::duration<double, std::milli> Duration;
typedef std::chrono::time_point<std::chrono::system_clock> TimePoint;

std::ostream& operator<<(std::ostream&, const glm::vec3&);
std::ostream& operator<<(std::ostream&, const glm::vec4&);

class Timer
{
public:
    Timer();
    ~Timer() {}
    void start();
    void stop();
    double getElapsedSeconds();
    double getElapsedMilliseconds();
    double getFrequency();
private:
    TimePoint start_point_;
    TimePoint end_point_;
};