from enum import Enum


class Color(Enum):
    Black = "black"
    Brown = " brown"
    Gray = "gray"
    Blue = "blue"
    Silver = "silver"
    White = "white"
    Pink = "pink"
    Purple = "purple"
    Red = "red"

class AgeGroup(Enum):
    child = "child"
    young = "young"
    old = "old"


class PersonGender(Enum):
    Male = "male"
    Female = "female"
    Other = "Other"


class PersonHairLength(Enum):
    LongHair = "long"
    ShortHair = "short"


class PersonClothingType(Enum):
    SimpleEveryday = "simple-everyday"
    Dignified = "dignified"
    Sporty = "sporty"


class PersonWearHeadgear(Enum):
    Hat = "hat"
    Cast = "cast"
    Cap = "cap"
    Scarf = "scarf"
    Keffiyeh = "keffiyeh"


class PersonShirtType(Enum):
    Long = "long"
    Short = "short"
    TankTop = "tank top"
    NoShirt = "no shirt"


class PersonPantsType(Enum):
    Sweatpants = "sweatpants"
    Jeans = "jeans"


class PersonPantsLongShort(Enum):
    Long = "long"
    short = "short "


class PersonGlassesType(Enum):
    SunGlasses = "sun glasses"
    EyeGlasses = "eye glasses"


class ShoesType(Enum):
    FlipFlops = "Flip flops"
    Sandals = "sandals"
    Sneakers = "sneakers"
    Boot = "boot"


class PersonShoesBrand(Enum):
    Nike = "nike"
    Vans = "vans"
    Adidas = "adidas"


class PersonOuterGarment(Enum):
    Coat = "Coat"
    raincoat = "raincoat"
    knitwear = "knitwear"
    hoodie = "hoodie"


class PersonHand(Enum):
    Left = "left"
    Right = "right"
    Both = "both"


class PersonStandingSitting(Enum):
    Standing = "standing"
    Sitting = 'sitting'


class PersonQuestion():
    class Visibility(Enum):
        ManOrFemale = f"Is the person is a male or female?"# {[member.value for member in PersonGender]}"
        AgeGroup = f"What is his age group" # - {[member.value for member in AgeGroup]}"
        IsExistsHair = "Does he have head hair?(yes or no)"
        IsBlackColorHair = "Is his color hair is black?(yes or no)"
        LongShorHair = f"Does he have long or short hair" # {[member.value for member in PersonHairLength]} "

    class Outfit(Enum):
        ClothingType = f"What is his clothing type" # {[member.value for member in PersonClothingType]}?"

        # ToDo fix he next paragraph
        IsOuterGarment = "Is he wearing an outer garment (yes or no)"
        IsOuterType = f"What outer garment he wear" # {[member.value for member in PersonOuterGarment]}?"
        OuterGarmentColor = f"What color his  outer garment " #{[member.value for member in Color]}"

        IsWearHeadgear = "Does the person wear any headgear? (Yes or no)"
        WearHeadgearType = f"what the person's headgear type" #{[member.value for member in PersonWearHeadgear]}"
        HeadgearColor = f"What the color of the person's headgear" # {[member.value for member in Color]} ? "

        IsHasBeard = "does he have a beard (yes or no) ? "

        IsWearFaceCovering = "Is he wearing a face covering? (yes or no)"
        FaceCoveringColor = f"What color is the face covering?" #{[member.value for member in Color]}"

        IsWearGlasses = "Does he wear glasses?(yes or no)?"  # YesNo
        GlassesType = f"what type of the glasses wear" # {[member.value for member in PersonGlassesType]}? "

        ShirtType = f"What kind of shirt does he wear"  # {[member.value for member in PersonShirtType]}"
        ShirtColor = f"What color is his shirt"  # {[member.value for member in Color]}?"

        IsWearingWatch = "Is he wearing a watch( yes or no)?"  # YesNo
        WatchColor = f"What is the color of his watch"  # {[member.value for member in Color]}?"

        IsWearingBelt = "Is he wearing a belt?"
        BeltColor = f"what is the color of his belt"  # {[member.value for member in Color]} ?"

        IsWearLongShortPants = f"Does he wear a short or long pants "  #{[member.value for member in PersonPantsLongShort]}? "
        PantsType = f"What type of the pants he wear"  # {[member.value for member in PersonPantsType]} ?  "
        PantsColor = f"What the color of the person's plants "  #{[member.value for member in Color]}?"

        IsWearShoes = "does he wear shoes?"
        ShoesType = "what type of his shoes"
        ShoesBrand = f"What brand his shoes"  # {[member.value for member in PersonShoesBrand]} ?"

    class Actions(Enum):
        IsHoldSomthing = "Is he holding something in his hand?(yes or no)"
        HandHoldWhichHand = f"In which hand he hold somthing "  #{[member.value for member in PersonHand]}"
        HandHoldType = "What does he hold?"

        IsEat = "Does the person eat?"

        IsSitting = f"Does the person sitting or standing "  #{[member.value for member in PersonStandingSitting]} "
