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
        weapon: document.getElementById("kingsud-weapon"),
        healtBar: document.getElementById("kingsud-health")
    };

    let playerTurn = true;

    function characterDied(character) {
        if (character.health <= 0) {
            setTimeout(() => {
                alert(character === npc ? "You Win!" : "You Lose!");
                window.location.href = "main_menu.html";
            }, 700);
        }
    }

    function updateHealth(character, damage) {
        character.health -= damage;
        if (character.health <= 0) {
            character.health = 0;
        }
        character.healtBar.style.width = character.health + "%";
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

    function npcTurn() {
        setTimeout(() => {
            if (npc.health > 0) {
                attack(npc, kingsud);
                updateHealth(kingsud, 10);
                playerTurn = true;
                document.getElementById("sword").disabled = false;
                characterDied(kingsud);
            }
        }, 500);
    }

    document.getElementById("sword").addEventListener("click", function () {
        if (playerTurn && npc.health > 0) {
            attack(kingsud, npc);
            updateHealth(npc, 10);
            playerTurn = false;
            document.getElementById("sword").disabled = true;
            npcTurn(); 
            characterDied(npc);
        }
    });
});
