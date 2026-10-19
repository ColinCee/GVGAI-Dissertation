package tracks;

import tracks.singlePlayer.advanced.sampleMCTS.SingleTreeNode;

import java.util.Random;

/**
 * Plays sampleMCTS on one game: RunMCTS gameFile levelPrefix levels gamesPerLevel seed.
 * Prints one CSV line per game: level,score,win,ticks,iters_per_move.
 * Use -Dgvgai.actionTime=40 -Dgvgai.maxTicks=2000 for 2018's competition settings.
 */
public class RunMCTS {
    public static void main(String[] args) {
        String game = args[0];
        String levelPrefix = args[1];
        String[] levels = args[2].split(",");
        int perLevel = Integer.parseInt(args[3]);
        Random seeds = new Random(Long.parseLong(args[4]));
        String agent = "tracks.singlePlayer.advanced.sampleMCTS.Agent";
        System.out.println("level,score,win,ticks,iters_per_move");
        for (String lvl : levels) {
            for (int i = 0; i < perLevel; i++) {
                long before = SingleTreeNode.TOTAL_ITERS;
                double[] r = ArcadeMachine.runOneGame(game, levelPrefix + lvl + ".txt", false, agent, null,
                        seeds.nextInt(), 0);
                long iters = SingleTreeNode.TOTAL_ITERS - before;
                int win = r[0] > 0 ? 1 : 0;
                System.out.printf("%s,%.1f,%d,%d,%.1f%n", lvl, r[1], win, (int) r[2], iters / Math.max(1.0, r[2]));
            }
        }
    }
}
