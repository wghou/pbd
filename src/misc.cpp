#include <limits>
#include "misc.h"

std::ostream&
operator<<(std::ostream& os, const glm::vec3& v)
{
    os << "(" << v.x << ", " << v.y << ", " << v.z << ")" << std::flush;
    return os;
}

std::ostream&
operator<<(std::ostream& os, const glm::vec4& v)
{
    os << "(" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ")" << std::flush;
    return os;
}


Timer::Timer()
{
    start_point_ = Clock::now();
    end_point_ = Clock::now();
}

void 
Timer::start()
{
    start_point_ = Clock::now();
}

void 
Timer::stop()
{
    end_point_ = Clock::now();
}

double 
Timer::getElapsedSeconds()
{
    Duration elapsed = end_point_ - start_point_;
    return elapsed.count() / 1000.0;
}

double
Timer::getElapsedMilliseconds()
{
    Duration elapsed = end_point_ - start_point_;
    return elapsed.count();
}

double
Timer::getFrequency()
{
    return 1.0 / (getElapsedSeconds() + std::numeric_limits<double>::epsilon());
}