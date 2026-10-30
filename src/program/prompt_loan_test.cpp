#include "strata/program/prompt_loan.hpp"

#include <cstdio>
#include <vector>

using namespace strata::program;

enum class Event { Quiesce, Refill, Relayout, Publish };

#define CHECK(cond)                                                                       \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::printf("prompt_loan_test FAIL line %d: %s\n", __LINE__, #cond);          \
            return 1;                                                                     \
        }                                                                                 \
        ++checks;                                                                         \
    } while (0)

int main() {
    int checks = 0;

    {
        std::vector<Event> got;
        bool quiesced = false;
        CHECK(prompt_loan_transition(
            quiesced, false,
            [&] { got.push_back(Event::Quiesce); },
            [&] { got.push_back(Event::Refill); return true; },
            [&] { got.push_back(Event::Relayout); return true; },
            [&] { got.push_back(Event::Publish); }));
        CHECK((got == std::vector<Event>{Event::Quiesce, Event::Relayout, Event::Publish}));
    }

    {
        std::vector<Event> got;
        bool quiesced = false;
        CHECK(prompt_loan_transition(
            quiesced, true,
            [&] { got.push_back(Event::Quiesce); },
            [&] { got.push_back(Event::Refill); return true; },
            [&] { got.push_back(Event::Relayout); return true; },
            [&] { got.push_back(Event::Publish); }));
        CHECK((got == std::vector<Event>{Event::Quiesce, Event::Refill,
                                         Event::Relayout, Event::Publish}));
    }

    {
        std::vector<Event> got;
        bool quiesced = false;
        CHECK(!prompt_loan_transition(
            quiesced, false,
            [&] { got.push_back(Event::Quiesce); },
            [&] { got.push_back(Event::Refill); return true; },
            [&] { got.push_back(Event::Relayout); return false; },
            [&] { got.push_back(Event::Publish); }));
        CHECK((got == std::vector<Event>{Event::Quiesce, Event::Relayout}));
    }

    {
        std::vector<Event> got;
        bool quiesced = false;
        auto run = [&] {
            return prompt_loan_transition(
                quiesced, false,
                [&] { got.push_back(Event::Quiesce); },
                [&] { got.push_back(Event::Refill); return true; },
                [&] { got.push_back(Event::Relayout); return true; },
                [&] { got.push_back(Event::Publish); });
        };
        CHECK(run());
        CHECK(run());
        CHECK((got == std::vector<Event>{Event::Quiesce, Event::Relayout, Event::Publish,
                                         Event::Relayout, Event::Publish}));
    }

    std::printf("prompt_loan_test OK (%d checks)\n", checks);
}
