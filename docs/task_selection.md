# Selected rollout tasks (108: 44 atomic + 64 composite)

Rule A (`scripts/select_tasks.py`): every pretrain atomic task, plus every pretrain composite task of a
RoboCasa task family that holds one of the 16 composite_unseen target tasks. Tasks whose demos replay to
success less than 80% of the time are dropped; the 50 target tasks are not candidates.

## Families of the composite_unseen tasks

| family | composite_unseen task | selected members |
|---|---|---|
| adding_ice_to_beverages | MakeIceLemonade | PlaceEqualIceCubes, PlaceIceInCup, RetrieveIceTray |
| arranging_cabinets | GatherTableware | ResetCabinetDoors, StackCans |
| arranging_condiments | CategorizeCondiments | LineUpCondiments, OrganizeCondiments |
| brewing | ArrangeTea | DeliverBrewedCoffee, OrganizeCoffeeCondiments, SweetenCoffee |
| chopping_vegetables | CuttingToolSelection | GatherCuttingTools |
| garnishing_dishes | GarnishPancake | AddLemonToFish, AddSugarCubes, GarnishCupcake |
| making_toast | BreadSelection | PrepareToast, SweetSavoryToastSetup |
| managing_freezer_space | SeparateFreezerRack | ClearFreezer, FreezeBottledWaters, FreezeIceTray, MaximizeFreezerSpace, MoveFridgeToFreezer, MoveToFreezerDrawer, ReorganizeFrozenVegetables |
| measuring_ingredients | WeighIngredients | ChooseMeasuringCup |
| organizing_recycling | RecycleBottlesByType | RecycleBottlesBySize, RecycleSodaCans, RecycleStackedYogurt |
| portioning_meals | PortionHotDogs | DistributeChicken, PortionFruitBowl, PortionInTupperware, PortionOnSize, PortionYogurt, ScalePortioning |
| preparing_sandwiches | HeatKebabSandwich | GatherVegetables, HotDogSetup, PrepareSausageCheese |
| reheating_food | WaffleReheat | HeatMug, MakeLoadedPotato, SimmeringSauce, WarmCroissant |
| serving_food | PanTransfer | AlcoholServingPrep, DessertUpgrade, PlaceFoodInBowls, PrepareSoupServing, ServeSteak |
| setting_the_table | ArrangeBreadBasket | AlignSilverware, ArrangeBreadBowl, ArrangeDrinkware, BeverageOrganization, DateNight, SeasoningSpiceSetup, SetBowlsForSoup, SetupBowls, SetupButterPlate, SetupFruitBowl, SetupWineGlasses, SizeSorting |
| washing_fruits_and_vegetables | WashFruitColander | AfterwashSorting, AirDryFruit, ClearClutter, ClearSinkSpace, DrainVeggies, PrewashFoodAssembly, PrewashFoodSorting |

## Dropped for demo replay

- PickPlaceCounterToBlender (atomic): 27%
- PickPlaceCounterToStandMixer (atomic): 25%
- TurnOnBlender (atomic): 8%

## All selected tasks

| task | kind | family | horizon | replay | pilot | goal |
|---|---|---|---|---|---|---|
| AddLemonToFish | composite | garnishing_dishes | 2100 | 100% | 1/2 | Take the lemon wedge from the fridge and place it on the plate with fish on the dining cou |
| AddSugarCubes | composite | garnishing_dishes | 900 | 100% | 2/2 | Take the two sugar cubes and place them on the plate with the cake. |
| AdjustToasterOvenTemperature | atomic | - | 450 | 100% | 2/2 | Decrease the toaster oven temperature. |
| AdjustWaterTemperature | atomic | - | 450 | 100% | 1/2 | The water running in the sink is cold. Adjust the faucet handle to run the water hot. Make |
| AfterwashSorting | composite | washing_fruits_and_vegetables | 3000 | 100% | 2/2 | Pick the broccolis from the sink and place them in one bowl. Place the tomato in the other |
| AirDryFruit | composite | washing_fruits_and_vegetables | 2850 | 100% | 2/2 | Pick the mango, mango from the sink and place them on the cutting board to airdry. Then tu |
| AlcoholServingPrep | composite | serving_food | 4650 | 100% | 2/2 | Open the cabinet directly in front. Then move the liquor and the cup to the dining counter |
| AlignSilverware | composite | setting_the_table | 2550 | 100% | 2/2 | Take the fork and place it on the left side of the plate. Then take the spoon and place it |
| ArrangeBreadBowl | composite | setting_the_table | 1350 | 100% | 1/2 | Pick up the heated bread from the toaster oven and place it in the bowl. Then move the bre |
| ArrangeDrinkware | composite | setting_the_table | 1650 | 100% | 2/2 | Place the pitcher and cup on the dining counter for serving. |
| BeverageOrganization | composite | setting_the_table | 4500 | 100% | 0/2 | Move the drinks to the dining counter. |
| CheesyBread | atomic | - | 750 | 100% | 2/2 | Pick up the wedge of cheese and place it on the slice of bread to prepare a simple cheese  |
| ChooseMeasuringCup | composite | measuring_ingredients | 1800 | 100% | 2/2 | Open the left drawer and place the smaller measuring cup on the counter. |
| ClearClutter | composite | washing_fruits_and_vegetables | 3300 | 100% | 2/2 | Pick up all produce items and place them in the sink. Turn on the sink faucet to wash them |
| ClearFreezer | composite | managing_freezer_space | 3600 | 100% | 2/2 | Take the chicken drumstick, peach, and peach from the freezer and place them in the bowl o |
| ClearSinkSpace | composite | washing_fruits_and_vegetables | 1800 | 100% | 2/2 | Pick the objects in the sink and place them on the counter to clear sink space for washing |
| CloseCabinet | atomic | - | 750 | 100% | 2/2 | Close the cabinet doors. |
| CloseDishwasher | atomic | - | 300 | 100% | 2/2 | Close the dishwasher door. |
| CloseDrawer | atomic | - | 450 | 100% | 2/2 | Close the right drawer. |
| CloseElectricKettleLid | atomic | - | 300 | 100% | 2/2 | Close the lid of the electric kettle. |
| CloseFridgeDrawer | atomic | - | 450 | 100% | 2/2 | Fully close the fridge drawer. |
| CloseMicrowave | atomic | - | 450 | 100% | 2/2 | Close the microwave door. |
| CloseOven | atomic | - | 450 | 100% | 2/2 | Close the oven door. |
| CloseStandMixerHead | atomic | - | 450 | 100% | 2/2 | Close the stand mixer head. |
| CoffeeServeMug | atomic | - | 450 | 100% | 0/2 | Pick the mug from under the coffee machine dispenser and place it on the counter. |
| DateNight | composite | setting_the_table | 4800 | 100% | 2/2 | Get the candle and the beer from the cabinet and move them to the dining counter. |
| DeliverBrewedCoffee | composite | brewing | 1050 | 100% | 1/2 | Pick up the brewed coffee from the coffee machine and place it next to the breakfast plate |
| DessertUpgrade | composite | serving_food | 1200 | 100% | 2/2 | Move the dessert items from the plates to the tray. |
| DistributeChicken | composite | portioning_meals | 2400 | 100% | 1/2 | Take the chicken drumsticks from the pan and place one on each plate on the dining counter |
| DrainVeggies | composite | washing_fruits_and_vegetables | 2400 | 100% | 2/2 | Dump the lemon from the pot into the sink. Then turn on the water and wash the lemon. Then |
| FreezeBottledWaters | composite | managing_freezer_space | 3900 | 100% | 1/2 | Take the bottled waters from the counter and place them in the freezer to freeze. Close th |
| FreezeIceTray | composite | managing_freezer_space | 2400 | 100% | 0/2 | The ice cube tray has been filled with water. Take the tray and place it in the freezer to |
| GarnishCupcake | composite | garnishing_dishes | 1800 | 100% | 2/2 | Take the cinnamon bottle from the cabinet and place it on the dining table, next to the pl |
| GatherCuttingTools | composite | chopping_vegetables | 2400 | 100% | 2/2 | Open the drawer on the left and place the cutting tools on the cutting board. |
| GatherVegetables | composite | preparing_sandwiches | 1350 | 100% | 2/2 | Pick up the bell pepper and eggplant from the sink and place them on the cutting board on  |
| HeatMug | composite | reheating_food | 1800 | 100% | 1/2 | Pick the mug from the counter and place it inside the microwave. Then close the microwave. |
| HotDogSetup | composite | preparing_sandwiches | 4200 | 100% | 0/2 |  |
| LineUpCondiments | composite | arranging_condiments | 3000 | 100% | 1/2 | Take two condiments from the cabinet and move them on the counter near the stove.Make sure |
| LowerHeat | atomic | - | 750 | 100% | 2/2 | Lower the heat of the kettle. |
| MakeIcedCoffee | atomic | - | 750 | 100% | 2/2 | Pick up an ice cube and place it in the glass of coffee. |
| MakeLoadedPotato | composite | reheating_food | 3000 | 100% | 2/2 | Retrieve the reheated potato from the microwave, then place the potato, cheese, and condim |
| MaximizeFreezerSpace | composite | managing_freezer_space | 1650 | 100% | 1/2 | Take the lemon and broccoli from the highest rack and move them to the second highest rack |
| MoveFridgeToFreezer | composite | managing_freezer_space | 1650 | 100% | 2/2 | There's a steak in the fridge that needs to be moved to the freezer for freezing. Pick up  |
| MoveToFreezerDrawer | composite | managing_freezer_space | 1800 | 100% | 2/2 | Take the broccoli and kiwi from the fridge shelf and place them in the freezer drawer to b |
| OpenBlenderLid | atomic | - | 450 | 100% | 2/2 | Open the blender by taking off the lid and placing it on the counter. |
| OpenDishwasher | atomic | - | 450 | 100% | 2/2 | Open the dishwasher door. |
| OpenElectricKettleLid | atomic | - | 300 | 100% | 2/2 | Press the button to open the lid of the electric kettle. |
| OpenFridge | atomic | - | 900 | 100% | 2/2 | Open the fridge door. |
| OpenFridgeDrawer | atomic | - | 450 | 100% | 2/2 | Fully open the fridge drawer. |
| OpenMicrowave | atomic | - | 600 | 100% | 2/2 | Open the microwave door. |
| OpenOven | atomic | - | 300 | 100% | 2/2 | Open the oven door. |
| OpenToasterOvenDoor | atomic | - | 450 | 100% | 2/2 | Open the toaster oven door. |
| OrganizeCoffeeCondiments | composite | brewing | 1950 | 100% | 2/2 | Pick the cinnamon and honey bottle from the cabinet and place them next to the mug. |
| OrganizeCondiments | composite | arranging_condiments | 2250 | 100% | 2/2 | There are various items on the counter. Only move the condiments/shakers into the cabinet. |
| PackDessert | atomic | - | 600 | 100% | 2/2 | Add the cupcake to the tupperware that contains the kebab skewer. |
| PickPlaceCabinetToCounter | atomic | - | 450 | 100% | 2/2 | Pick the croissant from the cabinet and place it on the counter. |
| PickPlaceCounterToDrawer | atomic | - | 750 | 100% | 2/2 | Pick the aluminum foil from the counter and place it in the drawer. |
| PickPlaceCounterToMicrowave | atomic | - | 1050 | 100% | 2/2 | Pick the broccoli from the counter and place it in the microwave. |
| PickPlaceCounterToOven | atomic | - | 600 | 100% | 2/2 | Place the corn on the rack of the oven. |
| PickPlaceCounterToSink | atomic | - | 600 | 100% | 2/2 | Pick the dish brush from the counter and place it in the sink. |
| PickPlaceCounterToToasterOven | atomic | - | 450 | 100% | 2/2 | Place the bread on the rack of the toaster oven. |
| PickPlaceFridgeDrawerToShelf | atomic | - | 600 | 100% | 2/2 | Pick the lemon wedge from the fridge drawer and place it on a fridge shelf. |
| PickPlaceFridgeShelfToDrawer | atomic | - | 600 | 100% | 2/2 | Pick the apple from the fridge shelf and place it in the fridge drawer. |
| PickPlaceMicrowaveToCounter | atomic | - | 750 | 100% | 2/2 | Pick the garlic from the microwave and place it on plate located on the counter. |
| PickPlaceStoveToCounter | atomic | - | 450 | 100% | 2/2 | Pick the garlic from the pan and place it in the bowl. |
| PickPlaceToasterOvenToCounter | atomic | - | 450 | 100% | 2/2 | Pick the bread from the rack and place it on the plate on the counter. |
| PlaceEqualIceCubes | composite | adding_ice_to_beverages | 2850 | 100% | 2/2 | There are four ice cubes in the ice bowl. Place two ice cubes in each glass of water. |
| PlaceFoodInBowls | composite | serving_food | 2550 | 100% | 2/2 | Pick both bowls and place them on the counter. Then place the mango in one bowl and place  |
| PlaceIceInCup | composite | adding_ice_to_beverages | 1200 | 100% | 2/2 | There are two ice cubes in the ice bowl. Place both ice cubes in the glass cup. |
| PortionFruitBowl | composite | portioning_meals | 2100 | 100% | 2/2 | Take the fruits from the plate on the dining counter and place two fruits in each bowl. |
| PortionInTupperware | composite | portioning_meals | 1950 | 100% | 2/2 | Place one cooked food and one vegetable from the large plate to each tupperware container. |
| PortionOnSize | composite | portioning_meals | 2700 | 100% | 1/2 | Take the smaller bowl and place it next to the appetizer plate on the dining counter, and  |
| PortionYogurt | composite | portioning_meals | 2100 | 100% | 2/2 | Place two yogurt containers on each plate. |
| PreheatOven | atomic | - | 450 | 100% | 2/2 | Preheat the oven by turning the temperature knob. |
| PrepareSausageCheese | composite | preparing_sandwiches | 3600 | 100% | 0/2 |  |
| PrepareSoupServing | composite | serving_food | 3450 | 100% | 2/2 | Open the cabinet and move the ladle to the pot. Then close the cabinet. |
| PrepareToast | composite | making_toast | 3750 | 100% | 2/2 | Pick the bread, place it on the cutting board, pick the jam, place it on the counter, and  |
| PrewashFoodAssembly | composite | washing_fruits_and_vegetables | 1650 | 100% | 2/2 | Pick the cucumber from the cabinet and place it in the bowl. Then pick the bowl and place  |
| PrewashFoodSorting | composite | washing_fruits_and_vegetables | 3600 | 100% | 1/2 | Pick the bananas from the cabinet and place them in one bowl. Pick the lemon wedge from th |
| RecycleBottlesBySize | composite | organizing_recycling | 3750 | 100% | 2/2 | Place the large bottle from the dining counter next to the other large bottle outside the  |
| RecycleSodaCans | composite | organizing_recycling | 6750 | 100% | 2/2 | Four empty soda cans are scattered around the kitchen. Gather all the cans and cluster the |
| RecycleStackedYogurt | composite | organizing_recycling | 2100 | 100% | 2/2 | Stack the yogurt containers by placing the smaller containers on top of the larger contain |
| ReorganizeFrozenVegetables | composite | managing_freezer_space | 2700 | 100% | 0/2 | Take the bell pepper and onion out of the freezer and place them in the tupperware contain |
| ResetCabinetDoors | composite | arranging_cabinets | 3300 | 100% | 2/2 | Close all open cabinet doors. |
| RetrieveIceTray | composite | adding_ice_to_beverages | 2550 | 100% | 1/2 | Open the freezer door, pick up the ice cube tray from the freezer, and place it on the din |
| ScalePortioning | composite | portioning_meals | 2100 | 100% | 2/2 | Take the chicken drumstick from the fridge and place it on the digital scale on the counte |
| SeasoningSpiceSetup | composite | setting_the_table | 4200 | 100% | 1/2 | Move the ketchup and mustard from the cabinet to the dining counter. |
| ServeSteak | composite | serving_food | 2100 | 100% | 1/2 | Pick up the pan with the steak in it and place it on the dining table. Then place the stea |
| SetBowlsForSoup | composite | setting_the_table | 5400 | 100% | 2/2 | Move the bowls from the cabinet to the plates on the dining table. |
| SetupBowls | composite | setting_the_table | 2700 | 100% | 1/2 | Pick up the bowls from the cabinet and place each bowl in front of a stool on the dining c |
| SetupButterPlate | composite | setting_the_table | 4050 | 100% | 0/2 |  |
| SetupFruitBowl | composite | setting_the_table | 3750 | 100% | 2/2 | Retrieve the peach and kiwi from the fridge and place them into the bowl on the dining tab |
| SetupWineGlasses | composite | setting_the_table | 3150 | 100% | 0/2 |  |
| SimmeringSauce | composite | reheating_food | 3450 | 100% | 2/2 | Place the pan on the front center burner on the stove. Then place the tomato and the onion |
| SizeSorting | composite | setting_the_table | 1500 | 100% | 2/2 | Stack the cups from largest to smallest. |
| SlideOvenRack | atomic | - | 450 | 100% | 2/2 | Fully slide the bottom oven rack in. |
| SlideToasterOvenRack | atomic | - | 300 | 100% | 1/2 | Fully slide the toaster oven rack out. |
| StackCans | composite | arranging_cabinets | 1800 | 100% | 2/2 | Stack two cans on top of the other two cans in the cabinet. |
| StartCoffeeMachine | atomic | - | 300 | 100% | 2/2 | Press the button on the coffee machine to serve coffee. |
| SweetSavoryToastSetup | composite | making_toast | 4050 | 100% | 2/2 | Pick the avocado and bread from the counter and place them on the plate. Then pick the jam |
| SweetenCoffee | composite | brewing | 1950 | 100% | 1/2 | Take a sugar cube from the saucer and place it inside the coffee. Then, grab milk from the |
| TurnOffMicrowave | atomic | - | 300 | 99% | 2/2 | Press the stop button on the microwave. |
| TurnOffSinkFaucet | atomic | - | 300 | 100% | 2/2 | Turn off the sink faucet. |
| TurnOnStove | atomic | - | 450 | 100% | 1/2 | Turn on the front center burner of the stove. |
| TurnOnToaster | atomic | - | 300 | 94% | 2/2 | Push down the lever of the toaster to turn it on. |
| TurnOnToasterOven | atomic | - | 450 | 100% | 2/2 | Turn on the toaster oven by setting the timer. |
| TurnSinkSpout | atomic | - | 300 | 100% | 2/2 | Turn the sink spout to the right. |
| WarmCroissant | composite | reheating_food | 1650 | 84% | 1/2 | Pick the croissant and place it on the pan. Then turn on the stove to warm the croissant. |
