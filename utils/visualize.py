import cv2


def overlay(display_image, category_index, boxes):

    box_color = (255, 128, 0)  # box color
    box_thickness = 2
    
    for b in boxes:
        cv2.rectangle(display_image, (b.x, b.y), (b.x+b.width, b.y+b.height), box_color, box_thickness)

        label_text = category_index[b.label]["name"]
        # draw the classification label string just above and to the left of the rectangle
        label_background_color = (0, int(b.score * 175), 75)
        label_text_color = (255, 255, 255)  # white text
        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        label_left = b.x
        label_top = b.y - label_size[1]
        
        if (label_top < 1):
            label_top = 1
        label_right = label_left + label_size[0]
        label_bottom = label_top + label_size[1]
        cv2.rectangle(display_image, (label_left - 1, label_top - 1), (label_right + 1, label_bottom + 1),
                    label_background_color, -1)

        # label text above the box
        cv2.putText(display_image, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)

    # display text to let user know how to quit
    cv2.rectangle(display_image,(0, 0),(100, 15), (128, 128, 128), -1)
    cv2.putText(display_image, "Press Q to Quit", (10, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    return display_image
