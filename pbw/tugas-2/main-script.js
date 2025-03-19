document.addEventListener("DOMContentLoaded", function () {
    let npc = { 
        health: 100, 
        element: document.getElementById("npc"), 
        weapon: document.getElementById("ops-weapon"),
        healtBar: document.getElementById("npc-health")
    };

    let kingsud = { 
        health: 100,
        element: document.getElementById("kingsud"),
        weapon: document.getElementById("kingsud-sword"),
        healtBar: document.getElementById("kingsud-health"),
        projectile: document.getElementById("kingsud-bow")
    };

    let playerTurn = true;

    function heal(character) {
        let chance = Math.floor(Math.random() * 10);
        if (chance < 5) {
            updateHealth(character, 10);
        }
        else if (chance > 5 && character === kingsud){
            updateHealth(character, 50);
        }
        else {
            updateHealth(character, 20);
        }
    }

    function characterDied(character) {
        if (character.health <= 0) {
            setTimeout(() => {
                alert(character === npc ? "You Win!" : "You Lose!");
                window.location.href = "main_menu.html";
            }, 700);
        }
    }

    function updateHealth(character, amount) {
        character.health += amount;
        if (character.health <= 0) {
            character.health = 0;
        }
        if (character.health > 100) {
            character.health = 100;
        }
        character.healtBar.style.width = character.health + "%";
        if (character.health < 20) {
            character.healtBar.style.backgroundColor = "#F93827";
        }
        else if (character.health < 60) {
            character.healtBar.style.backgroundColor = "#FF9D23";
        }
        else {
            character.healtBar.style.backgroundColor = "#16C47F";
        }
    }

    function attack(attacker, defender) {
        attacker.weapon.style.animation = "none";
        void attacker.weapon.offsetWidth;
        attacker.weapon.style.animation = attacker === kingsud ? "throw-right 0.3s ease-out" : "throw-left 0.3s ease-out";
        defender.element.classList.add("shake");
        defender.weapon.classList.add("shake");
        setTimeout(() => defender.element.classList.remove("shake"), 250);
        setTimeout(() => defender.Weapon.classList.remove("shake"), 250);
    }

    function projectile(attacker, defender, hit) {
        attacker.projectile.style.animation = "none";
        void attacker.projectile.offsetWidth;
        attacker.projectile.style.animation = attacker === kingsud ? "projectile-right 0.3s linear" : "projectile-left 0.6s ease-in-out";
        if (hit) {
            defender.element.classList.add("shake");
            defender.weapon.classList.add("shake");
            setTimeout(() => defender.element.classList.remove("shake"), 250);
            setTimeout(() => defender.Weapon.classList.remove("shake"), 250);
        }
    }

    function npcTurn() {
        setTimeout(() => {
            if (npc.health > 0) {
                let chance = Math.floor(Math.random() * 10);
                if (chance < 8) {
                    attack(npc, kingsud);
                    updateHealth(kingsud, -5);
                    playerTurn = true;
                    characterDied(kingsud);
                }
                else {
                    heal(npc);
                    playerTurn = true;
                    characterDied(kingsud);
                }
            }
        }, 500);
    }

    document.getElementById("sword").addEventListener("click", function () {
        if (playerTurn && npc.health > 0) {
            attack(kingsud, npc);
            updateHealth(npc, -10);
            playerTurn = false;
            npcTurn(); 
            characterDied(npc);
        }
    });

    document.getElementById("bow").addEventListener("click", function () {
        if (playerTurn && npc.health > 0) {
            let chance = Math.floor(Math.random() * 10);
            let hit = chance > 5 ? true : false;
            projectile(kingsud, npc, hit);
            if (hit) {
                updateHealth(npc, -20);   
            }
            playerTurn = false;
            npcTurn(); 
            characterDied(npc);
        }
    });

    document.getElementById("heal").addEventListener("click", function () {
        if (playerTurn && npc.health > 0 && kingsud.health < 100) {
            heal(kingsud);
            playerTurn = false;
            npcTurn(); 
            characterDied(npc);
        }
    });

    document.getElementById("do-nothing").addEventListener("click", function () {
        if (playerTurn && npc.health > 0 && kingsud.health < 100) {
            playerTurn = false;
            npcTurn(); 
            characterDied(npc);
        }
    });
});
