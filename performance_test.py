import cProfile
import pstats
import game


def profile_game():
    profiler = cProfile.Profile()
    profiler.enable()
    game.main()
    profiler.disable()

    # Save profiling data
    with open("profile_results.txt", "w") as file:
        stats = pstats.Stats(profiler, stream=file)
        stats.sort_stats("cumtime")
  # Sort by cumulative time
        stats.print_stats()


if __name__ == "__main__":
    profile_game()
