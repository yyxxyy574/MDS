import pygame
import pytmx
import re
from openai import OpenAI
import os

from .character import Character, ARROW_DICT

MODEL = "gpt-4.1-mini-2025-04-14"
REGION = "eastus2"
API_BASE = "https://api.openai.com"
ENDPOINT = f"{API_BASE}/{REGION}"

class Canvas:
    def __init__(self, tileset_character_path, tileset_animal_path, tileset_arrow_path, tileset_emoji_path, tileset_emoji_box_path, tmx_path=None, image_path=None):
        # Init pygame
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        # Load tmx
        self.tmx_data = None
        if tmx_path is not None:
            self.tmx_data = pytmx.load_pygame(tmx_path)
            self.tile_w, self.tile_h = self.tmx_data.tilewidth, self.tmx_data.tileheight
            self.map_w, self. map_h = self.tmx_data.width * self.tile_w, self.tmx_data.height * self.tile_h
        # Load image
        self.image = None
        if image_path is not None:
            self.image = pygame.image.load(image_path)
            self.tile_w, self.tile_h = 48, 48
            self.map_w, self. map_h = self.image.get_width(), self.image.get_height()
        # Load tileset
        self.tileset_character = pygame.image.load(tileset_character_path).convert_alpha()
        self.tileset_character.set_colorkey((0, 0, 0))
        self.tileset_animal = pygame.image.load(tileset_animal_path).convert_alpha()
        self.tileset_animal.set_colorkey((0, 0, 0))
        self.tileset_arrow = pygame.image.load(tileset_arrow_path).convert_alpha()
        self.tileset_emoji = pygame.image.load(tileset_emoji_path).convert_alpha()
        self.tileset_emoji.set_colorkey((0, 0, 0))
        self.tileset_emoji_box = pygame.image.load(tileset_emoji_box_path).convert_alpha()
        self.tileset_emoji_box.set_colorkey((0, 0, 0))
        # Init layers
        self.layers = {}
        # Text
        self.description_h = 1000
        self.padding = 20
        self.font = pygame.font.Font(None, 24)
        self.line_h = self.font.get_height() + 5

        # openai client
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            max_retries=10,
        )

    def load_map(self):
        for layer in self.tmx_data.visible_layers:
            if isinstance(layer, pytmx.TiledTileLayer):
                surface = pygame.Surface((self.map_w, self.map_h), pygame.SRCALPHA)
                self.layers[layer.name] = surface
                for x, y, gid in layer:
                    tile = self.tmx_data.get_tile_image_by_gid(gid)
                    if tile:
                        surface.blit(tile, (x * self.tile_w, y * self.tile_h))
        
        surface = pygame.Surface((self.map_w, self.map_h), pygame.SRCALPHA)
        self.layers['arrow'] = surface
        surface = pygame.Surface((self.map_w, self.map_h), pygame.SRCALPHA)
        self.layers['ui'] = surface

    def load_ui(self, ui_pos, top_pos, is_box=True, is_human=False, is_animal=False):
        surface = self.layers['ui']
        if is_human:
            src = pygame.Rect(ui_pos[0] * self.tile_w, ui_pos[1] * self.tile_h, self.tile_w, self.tile_h)
            dest = pygame.Rect(top_pos[0] - (self.tile_w // 2), top_pos[1], self.tile_w, self.tile_h)
            surface.blit(self.tileset_character, dest, src)
        elif is_animal:
            src = pygame.Rect(ui_pos[0] * self.tile_w, ui_pos[1] * self.tile_h, self.tile_w, self.tile_h)
            dest = pygame.Rect(top_pos[0] - (self.tile_w // 2), top_pos[1], self.tile_w, self.tile_h)
            surface.blit(self.tileset_animal, dest, src)
        elif is_box:
            src = pygame.Rect(ui_pos[0] * self.tile_w, ui_pos[1] * self.tile_h * 2, self.tile_w, self.tile_h * 2)
            dest = pygame.Rect(top_pos[0] - (self.tile_w // 2), top_pos[1], self.tile_w, self.tile_h * 2)
            surface.blit(self.tileset_emoji_box, dest, src)
        else:
            src = pygame.Rect(ui_pos[0] * self.tile_w, ui_pos[1] * self.tile_h, self.tile_w, self.tile_h)
            dest = pygame.Rect(top_pos[0] - (self.tile_w // 2), top_pos[1], self.tile_w, self.tile_h)
            surface.blit(self.tileset_emoji, dest, src)

    def load_character(self, character: Character, layer='object'):
        surface = self.layers[layer]
        if character.species == 'human':
            dest = pygame.Rect(character.lower_body.x * self.tile_w, character.lower_body.y * self.tile_h, character.width, character.height)
            surface.blit(self.tileset_character, dest, character.lower_body.src_rect)
            dest = pygame.Rect(character.upper_body.x * self.tile_w, character.upper_body.y * self.tile_h, character.width, character.height)
            surface.blit(self.tileset_character, dest, character.upper_body.src_rect)
        else:
            dest = pygame.Rect(character.body.x * self.tile_w, character.body.y * self.tile_h, character.width, character.height)
            surface.blit(self.tileset_animal, dest, character.body.src_rect)
        
        if character.arrow:
            surface = self.layers['arrow']
            dest = pygame.Rect(character.arrow.x * self.tile_w, character.arrow.y * self.tile_h, character.width, character.height)
            surface.blit(self.tileset_arrow, dest, character.arrow.src_rect)

    def load_text(self, template, text_info, scope, layer='object'):
        try:
            text = template.format(**text_info)
        except KeyError as e:
            print(f"  Warning: Missing key {e} in {text_info} for template.")
            text = template
        lines = text.split("\n")

        surface = self.layers[layer]
        rect = pygame.Rect(scope[0][0] * self.tile_w, scope[0][1] * self.tile_h, (scope[1][0] - scope[0][0] + 1) * self.tile_w, (scope[1][1] - scope[0][1] + 1) * self.tile_h)
        
        font_size = 50
        while True:
            font = pygame.font.Font(None, font_size)
            max_line_w = max(font.size(line)[0] for line in lines)
            if max_line_w > rect.width or font.get_height() * len(lines) > rect.height:
                font_size -= 1
                if font_size < 10:
                    break
            else:
                break
        
        y_offset = rect.top
        for line in lines:
            text_surface = font.render(line, True, (0, 0, 0))
            text_w, text_h = text_surface.get_size()

            x_offset = rect.left + (rect.width - text_w) // 2
            surface.blit(text_surface, (x_offset, y_offset))
            y_offset += text_h

        return [rect.left + rect.width // 2, y_offset]

    def load_description(self, template, character_description, rewrite=True):
        surface = pygame.Surface((self.map_w, self.map_h + self.description_h), pygame.SRCALPHA)
        self.layers['description'] = surface

        try:
            for key, value in character_description.items():
                if isinstance(value, str) and value.endswith(", "):
                    character_description[key] = value[:-2]
            formatted_description = template.format(**character_description)
        except KeyError as e:
            print(f"  Warning: Missing key {e} in {character_description} for template.")
            formatted_description = template
        
        if rewrite:
            response = self.client.responses.create(
                model=MODEL,
                input=f"This is a description of a moral dilemma:\n\
                        {formatted_description}\n\
                        Rewrite this description as one paragraph to make it more fluent, natural, concise and understandable. Merge and arrange the lists of characters in the parentheses (for example, 'female doctor, female doctor, sheep, female child human' into 'two female doctor, a girl and a sheep'), adapt characteristics of each character (for example, 'yellow male elderly eastern' into 'an old yellow male from the east'), delete something like 0 species, but do not remove the given characteristics. If the merged list exceeds five entries, only list five and add an ellipsis. The last sentence should maintain the form of yes or not question. Keep every '|| ||' in the original position, do not add any, and do not change the content in '|| ||'. Provide the modified description directly:",
                temperature=0
            )
            modified_description = response.output[0].content[0].text.strip()
        else:
            modified_description = formatted_description

        parts = re.split(r'(\|\|ARROW:.*?\|\|)', modified_description)

        x, y = self.padding, self.map_h + 10

        for part in parts:
            if not part:
                continue

            if part.startswith("||ARROW:"):
                color = part.strip().replace('||ARROW: ', '').replace('||', '').replace('{', '').replace('}', '')

                if color in ARROW_DICT:
                    pose_arrow_x = ARROW_DICT[color]
                    arrow_rect = pygame.Rect(pose_arrow_x * self.tile_w, 0, self.tile_w, self.tile_h)
                    
                    if x + self.tile_w > self.map_w - self.padding:
                        x = self.padding
                        y += self.line_h

                    arrow_img_scaled = pygame.transform.scale(self.tileset_arrow.subsurface(arrow_rect), (self.padding, self.padding))
                    surface.blit(arrow_img_scaled, (x, y + (self.font.get_height() - arrow_img_scaled.get_height()) // 2))
                    
                    x += arrow_img_scaled.get_width() + 5
                else:
                    print(f"  Warning: Arrow color '{color}' not found.")
            
            else:
                words = part.split(" ")
                for word in words:
                    if not word:
                        continue
                    word_surface = self.font.render(word, True, (0, 0, 0))
                    word_w, word_h = word_surface.get_size()

                    if x + word_w >= self.map_w - self.padding:
                        x = self.padding
                        y += self.line_h
                    
                    surface.blit(word_surface, (x, y))
                    x += word_w + self.font.size(" ")[0]

        self.description_h = y + 20

        return modified_description

    def render(self, save_path, load_description=True, show=False):
        if load_description:
            self.screen = pygame.display.set_mode((self.map_w, self.description_h))
            self.screen.fill((255, 255, 255))
            self.screen.blit(self.layers['description'], (0, 0))
        else:
            self.screen = pygame.display.set_mode((self.map_w, self.map_h))

        if self.tmx_data is not None:
            for layer in self.tmx_data.visible_layers:
                self.screen.blit(self.layers[layer.name], (0, 0))
            self.screen.blit(self.layers['ui'], (0, 0))
            self.screen.blit(self.layers['arrow'], (0, 0))

        if self.image is not None:
            self.screen.blit(self.image, (0, 0))

        pygame.display.flip()
        pygame.image.save(self.screen, save_path)

        if show:
            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
            pygame.quit()

    def reset(self):
        self.layers = {}
        self.load_map()
