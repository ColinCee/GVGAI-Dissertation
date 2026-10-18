package core.vgdl;

import core.competition.CompetitionParameters;
import core.game.Game;
import core.player.LearningPlayer;
import core.player.Player;
import ontology.Types;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

/**
 * Created with IntelliJ IDEA.
 * User: Diego
 * Date: 24/10/13
 * Time: 10:54
 * This is a Java port from Tom Schaul's VGDL - https://github.com/schaul/py-vgdl
 */
public class VGDLViewer extends JComponent
{
    /**
     * Reference to the game to be painted.
     */
    public Game game;

    /**
     * Dimensions of the window.
     */
    private Dimension size;

    /**
     * Sprites to draw
     */
    public SpriteGroup[] spriteGroups;

    /**
     * Player of the game
     */
    public Player player;

    public boolean justImage = false;

    /**
     * Creates the viewer for the game.
     * @param game game to be displayed
     */
    public VGDLViewer(Game game, Player player)
    {
        this.game = game;
        this.size = game.getScreenSize();
        this.player = player;
        if (player instanceof LearningPlayer) {
            LearningPlayer learningPlayer = (LearningPlayer) player;
            Types.LEARNING_SSO_TYPE ssoType = learningPlayer.getLearningSsoType();
            if (ssoType == Types.LEARNING_SSO_TYPE.IMAGE ||
                ssoType == Types.LEARNING_SSO_TYPE.BOTH) {
                saveImage(CompetitionParameters.SCREENSHOT_FILENAME);
            }
        }
    }

    /**
     * Main method to paint the game
     * @param gx Graphics object.
     */
    public void paintComponent(Graphics gx)
    {
        Graphics2D g = (Graphics2D) gx;
        paintWithGraphics(g);
    }

    public void paintWithGraphics(Graphics2D g) {
        //For a better graphics, enable this: (be aware this could bring performance issues depending on your HW & OS).
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        //g.setColor(Types.LIGHTGRAY);
        g.setColor(Types.BLACK);
        g.fillRect(0, size.height, size.width, size.height);

        try {
            int[] gameSpriteOrder = game.getSpriteOrder();
            if (this.spriteGroups != null) for (Integer spriteTypeInt : gameSpriteOrder) {
                if (spriteGroups[spriteTypeInt] != null) {
                    ArrayList<VGDLSprite> spritesList = spriteGroups[spriteTypeInt].getSprites();
                    for (VGDLSprite sp : spritesList) {
                        if (sp != null) sp.draw(g, game);
                    }

                }
            }
        }catch(Exception e) {}

        g.setColor(Types.BLACK);
        player.draw(g);
    }



    /**
     * Paints the sprites.
     * @param spriteGroupsGame sprites to paint.
     */
    public void paint(SpriteGroup[] spriteGroupsGame)
    {
        if (RAW_FRAMES && justImage) {
            // Headless: nothing draws concurrently, so draw the game's own sprites instead of a copy.
            this.spriteGroups = spriteGroupsGame;
            saveImage(CompetitionParameters.SCREENSHOT_FILENAME);
            return;
        }
        this.spriteGroups = new SpriteGroup[spriteGroupsGame.length];
        for(int i = 0; i < this.spriteGroups.length; ++i)
        {
            this.spriteGroups[i] = new SpriteGroup(spriteGroupsGame[i].getItype());
            this.spriteGroups[i].copyAllSprites(spriteGroupsGame[i].getSprites());
        }
        this.repaint();
        if (player instanceof LearningPlayer) {
            LearningPlayer learningPlayer = (LearningPlayer) player;
            Types.LEARNING_SSO_TYPE ssoType = learningPlayer.getLearningSsoType();

            if (ssoType == Types.LEARNING_SSO_TYPE.IMAGE ||
                ssoType == Types.LEARNING_SSO_TYPE.BOTH) {
                saveImage(CompetitionParameters.SCREENSHOT_FILENAME);
            }
        }
    }

    /**
     * Gets the dimensions of the window.
     * @return the dimensions of the window.
     */
    public Dimension getPreferredSize() {
        return size;
    }

    public static byte[] lastPng = null;

    // -Dgvgai.rawFrames=true: keep the frame as RGB bytes (alpha dropped, as the client's PNG->RGB did)
    // and skip PNG encoding and the frame file. Off by default, so the original protocol still works.
    public static final boolean RAW_FRAMES = Boolean.getBoolean("gvgai.rawFrames");
    public static byte[] lastRgb = null;
    public static int lastWidth, lastHeight;

    private BufferedImage rawImage;  // RAW_FRAMES: one image and graphics context reused for every frame
    private Graphics2D rawGraphics;

    public void saveImage(String fileName)  {
        try {
            if (RAW_FRAMES) {
                if (rawImage == null) {
                    rawImage = new BufferedImage((int) size.getWidth(), (int) size.getHeight(), BufferedImage.TYPE_INT_ARGB);
                    rawGraphics = rawImage.createGraphics();
                    lastRgb = new byte[rawImage.getWidth() * rawImage.getHeight() * 3];
                }
                int[] argb = ((java.awt.image.DataBufferInt) rawImage.getRaster().getDataBuffer()).getData();
                java.util.Arrays.fill(argb, 0);  // as a fresh image: transparent black
                paintWithGraphics(rawGraphics);
                // lastRgb is reused: each frame is sent before the next one is drawn.
                byte[] rgb = lastRgb;
                for (int i = 0, j = 0; i < argb.length; i++) {
                    int p = argb[i];
                    rgb[j++] = (byte) (p >> 16);
                    rgb[j++] = (byte) (p >> 8);
                    rgb[j++] = (byte) p;
                }
                lastWidth = rawImage.getWidth();
                lastHeight = rawImage.getHeight();
                return;
            }
            BufferedImage bi = new BufferedImage( (int) size.getWidth(), (int) size.getHeight(), BufferedImage.TYPE_INT_ARGB);
            Graphics2D graphics = bi.createGraphics();
            paintWithGraphics(graphics);
            File file = new File(fileName);
            if (justImage) {
                graphics.dispose();
            }
            // Encode once and keep the bytes, so the state observation needn't re-read the file.
            java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
            ImageIO.write(bi, "png", out);
            lastPng = out.toByteArray();
            java.nio.file.Files.write(file.toPath(), lastPng);
        } catch (IOException ie) {
            ie.printStackTrace();
        }

    }
}
